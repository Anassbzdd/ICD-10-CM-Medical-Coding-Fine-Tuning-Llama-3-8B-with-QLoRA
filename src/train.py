"""QLoRA training entrypoint for Meta-Llama-3-8B on cleaned ICD-10 data."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
import wandb
from datasets import load_from_disk, DatasetDict, load_dataset
from transformers import TrainerCallback , TrainingArguments

try:
    from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
except ImportError:
    from trl import SFTTrainer
    from trl.trainer.utils import DataCollatorForCompletionOnlyLM

from src.config import ModelConfig, DataConfig, TrainingConfig
from src.modeling import get_tokenizer, load_model_for_training
from src.preprocessing import build_dataset_report, materialize_clean_dataset, split_dataset
from src.utils import (
    detect_compute_environment, 
    ensure_dir, 
    get_logger, 
    resolve_resume_checkpoint, 
    save_json, seed_everything, 
    setup_logging, 
    to_serializable
)

LOGGER = get_logger(__name__)

class TrainableParameterCallback(TrainerCallback):
    """Log trainable parameter counts at training start for quick sanity checks."""

    def on_train_begin(self, args, state, control,model = None, **kwargs):
        if model is not None and hasattr(model, "print_trainable_parameters"):
            model.print_trainable_parameters()

def load_or_prepare_dataset(
        model_config: ModelConfig,
        data_config: DataConfig
) -> DatasetDict:
    """Load a saved processed dataset, or build it on the fly when missing."""

    if data_config.processed_dir.exists():
        LOGGER.info("Loading processed dataset from %s.", data_config.processed_dir)
        return load_from_disk(str(data_config.processed_dir))
    
    LOGGER.warning(
        "Processed dataset not found at %s. Building it inline so training can proceed.",
        data_config.processed_dir,
    )
    tokenizer = get_tokenizer(model_config)
    raw_dataset_dict = load_dataset(data_config.dataset_name,name = data_config.dataset_config_name)
    raw_split = raw_dataset_dict[data_config.raw_split]
    clean_dataset, stats = materialize_clean_dataset(raw_split, tokenizer)
    split_dataset_dict = split_dataset(clean_dataset,data_config)
    report = build_dataset_report(
        raw_dataset= raw_split,
        clean_dataset=clean_dataset,
        split_dataset_dict=split_dataset_dict,
        tokenizer=tokenizer,
        stats = stats,
        sample_size=data_config.sample_report_size,
    )
    split_dataset_dict.save_to_disk(str(data_config.processed_dir))
    save_json(report, data_config.report_path)
    return split_dataset_dict

def maybe_limit_dataset(dataset, max_samples: int | None):
    """Optionally truncate a dataset for quick debugging or constrained runs."""

    if max_samples is None:
        return dataset
    return dataset.select(range(min(len(dataset), max_samples)))

def validate_response_template(tokenizer , sample_text:str, response_template:str):
    sample_ids = tokenizer(sample_text, add_special_tokens= False)["input_ids"]
    template_ids = tokenizer(response_template, add_special_tokens= False)["input_ids"]
    if not template_ids:
        raise ValueError("Response template tokenization produced an empty token sequence.")
    for start_index in range(0, len(sample_ids) - len(template_ids) + 1):
        if sample_ids[start_index: start_index + len(template_ids)] == template_ids:
            return
    raise ValueError(
        "Response template was not found in the tokenized training sample. "
        "Completion-only loss masking would fail with the current prompt format."
    )

def build_training_arguments(training_config : TrainingConfig) -> TrainingArguments:
    """Create Hugging Face training arguments for QLoRA fine-tuning."""

    bf16_supported = bool(torch.cuda.is_available() and torch.cuda.is_bf16_supported())
    if not bf16_supported:
        LOGGER.warning(
            "Current GPU does not report bf16 support. Keeping 4-bit compute dtype at bf16 as requested, "
            "but trainer mixed precision will use fp16 for compatibility."
        )
    ensure_dir(training_config.output_dir)
    return TrainingArguments(
        output_dir=str(training_config.output_dir),
        run_name=training_config.run_name,
        num_train_epochs=training_config.num_train_epochs,
        per_device_train_batch_size=training_config.per_device_train_batch_size,
        per_device_eval_batch_size=training_config.per_device_eval_batch_size,
        gradient_accumulation_steps=training_config.gradient_accumulation_steps,
        learning_rate=training_config.learning_rate,
        warmup_ratio=training_config.warmup_ratio,
        weight_decay=training_config.weight_decay,
        lr_scheduler_type=training_config.lr_scheduler_type,
        optim=training_config.optim,
        logging_steps=training_config.logging_steps,
        evaluation_strategy="steps",
        eval_steps=training_config.eval_steps,
        save_strategy="steps",
        save_steps=training_config.save_steps,
        save_total_limit=training_config.save_total_limit,
        max_grad_norm=training_config.max_grad_norm,
        gradient_checkpointing=training_config.gradient_checkpointing,
        fp16=not bf16_supported,
        bf16=bf16_supported,
        report_to=["wandb"] if training_config.report_to_wandb else [],
        ddp_find_unused_parameters=training_config.ddp_find_unused_parameters,
        dataloader_pin_memory=True,
        remove_unused_columns=False,
        logging_first_step=True,
        group_by_length=True,
        )

def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for model training."""

    data_defaults = DataConfig()
    model_defaults = ModelConfig()
    train_defaults = TrainingConfig()

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed_dir", type=Path, default=data_defaults.processed_dir)
    parser.add_argument("--dataset_name", type=str, default=data_defaults.dataset_name)
    parser.add_argument("--dataset_config_name", type=str, default=data_defaults.dataset_config_name)
    parser.add_argument("--raw_split", type=str, default=data_defaults.raw_split)
    parser.add_argument("--report_path", type=Path, default=data_defaults.report_path)
    parser.add_argument("--sample_report_size", type=int, default=data_defaults.sample_report_size)
    parser.add_argument("--seed", type=int, default=data_defaults.seed)
    parser.add_argument("--model_name", type=str, default=model_defaults.model_name)
    parser.add_argument("--tokenizer_name", type=str, default=model_defaults.tokenizer_name)
    parser.add_argument("--cache_dir", type=str, default=model_defaults.cache_dir)
    parser.add_argument("--max_seq_length", type=int, default=model_defaults.max_seq_length)
    parser.add_argument("--lora_r", type=int, default=model_defaults.lora_r)
    parser.add_argument("--lora_alpha", type=int, default=model_defaults.lora_alpha)
    parser.add_argument("--lora_dropout", type=float, default=model_defaults.lora_dropout)
    parser.add_argument("--output_dir", type=Path, default=train_defaults.output_dir)
    parser.add_argument("--run_name", type=str, default=train_defaults.run_name)
    parser.add_argument("--per_device_train_batch_size", type=int, default=train_defaults.per_device_train_batch_size)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=train_defaults.per_device_eval_batch_size)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=train_defaults.gradient_accumulation_steps)
    parser.add_argument("--num_train_epochs", type=float, default=train_defaults.num_train_epochs)
    parser.add_argument("--learning_rate", type=float, default=train_defaults.learning_rate)
    parser.add_argument("--warmup_ratio", type=float, default=train_defaults.warmup_ratio)
    parser.add_argument("--weight_decay", type=float, default=train_defaults.weight_decay)
    parser.add_argument("--logging_steps", type=int, default=train_defaults.logging_steps)
    parser.add_argument("--eval_steps", type=int, default=train_defaults.eval_steps)
    parser.add_argument("--save_steps", type=int, default=train_defaults.save_steps)
    parser.add_argument("--save_total_limit", type=int, default=train_defaults.save_total_limit)
    parser.add_argument("--resume_from_checkpoint", type=str, default=train_defaults.resume_from_checkpoint)
    parser.add_argument("--max_train_samples", type=int, default=train_defaults.max_train_samples)
    parser.add_argument("--max_eval_samples", type=int, default=train_defaults.max_eval_samples)
    parser.add_argument("--wandb_project", type=str, default=train_defaults.wandb_project)
    parser.add_argument("--wandb_entity", type=str, default=train_defaults.wandb_entity)
    parser.add_argument("--no_wandb", action="store_true")
    parser.add_argument("--log_level", type=str, default=train_defaults.log_level)
    return parser.parse_args()


def main() -> None:
    """Run supervised fine-tuning with completion-only loss masking."""

    args = parse_args()
    setup_logging(args.log_level)
    
    data_config = DataConfig(
        dataset_name = args.dataset_name,
        dataset_config_name = args.dataset_config_name,
        raw_split = args.raw_split,
        seed = args.seed,
        processed_dir = args.processed_dir ,
        report_path = args.report_path,
        sample_report_size= args.sample_report_size,
    )
    model_config = ModelConfig(
        model_name = args.model_name,
        tokenizer_name = args.tokenizer_name,
        cache_dir = args.cache_dir,
        max_seq_length = args.max_seq_length,
        lora_r = args.lora_r,
        lora_alpha = args.lora_alpha,
        lora_dropout = args.lora_dropout,
    )
    training_config = TrainingConfig(
        output_dir=args.output_dir,
        run_name=args.run_name,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        logging_steps=args.logging_steps,
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        resume_from_checkpoint=args.resume_from_checkpoint,
        max_train_samples=args.max_train_samples,
        max_eval_samples=args.max_eval_samples,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        report_to_wandb=not args.no_wandb,
        log_level=args.log_level,
    )
    seed_everything(data_config.seed)
    LOGGER.info("Hardware summary: %s", detect_compute_environment())

    dataset_dict = load_or_prepare_dataset(model_config=model_config,data_config=data_config)
    train_dataset = maybe_limit_dataset(dataset_dict["train"], training_config.max_train_samples)
    eval_dataset = maybe_limit_dataset(dataset_dict["validation"], training_config.max_eval_samples)
    LOGGER.info("Training rows: %d | Validation rows: %d", len(train_dataset), len(eval_dataset))

    if training_config.report_to_wandb:
        os.environ["WANDB_PROJECT"] = training_config.wandb_project
        if training_config.wandb_entity:
            os.environ["WANDB_ENTITY"] = training_config.wandb_entity
        wandb.init(
            project = training_config.wandb_project,
            entity= training_config.wandb_entity,
            name = training_config.run_name,
            config = {
                "model": to_serializable(model_config),
                "data": to_serializable(data_config),
                "training": to_serializable(training_config),
            },
        )

    tokenizer = get_tokenizer(model_config)
    validate_response_template(tokenizer , train_dataset[0]["text"] , model_config.response_template)
    model = load_model_for_training(model_config)
    data_collator = DataCollatorForCompletionOnlyLM(
        response_template = model_config.response_template,
        tokenizer = tokenizer,
    )

    training_args = build_training_arguments(training_config)
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        dataset_text_field="text",
        tokenizer=tokenizer,
        data_collator=data_collator,
        max_seq_length=model_config.max_seq_length,
        packing=False,
        callbacks=[TrainableParameterCallback()],
    )

    resume_checkpoint = resolve_resume_checkpoint(training_config.output_dir, training_config.resume_from_checkpoint)
    if resume_checkpoint:
        LOGGER.info("Resuming training from checkpoint %s.", resume_checkpoint)
    train_result = trainer.train(resume_from_checkpoint=resume_checkpoint)

    trainer.save_model()
    tokenizer.save_pretrained(str(training_config.output_dir))
    metrics = dict(train_result.metrics)
    metrics["train_dataset_size"] = len(train_dataset)
    metrics["eval_dataset_size"] = len(eval_dataset)
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    if training_config.report_to_wandb and wandb.run is not None:
        wandb.log(metrics)
        wandb.finish()

if __name__ == "__main__":
    main()