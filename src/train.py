"""QLoRA training entrypoint for Meta-Llama-3-8B on cleaned ICD-10 data."""

from __future__ import annotations

import argparse
import inspect
import os
from pathlib import Path

import torch
import wandb
from datasets import load_from_disk, DatasetDict, load_dataset
from huggingface_hub import whoami
from transformers import TrainerCallback

from trl import SFTConfig, SFTTrainer

from src.config import LLAMA3_EOT_TOKEN, ModelConfig, DataConfig, TrainingConfig
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


def add_prompt_completion_columns(dataset):
    """Expose prompt/completion fields for modern TRL completion-only loss."""

    def convert(example):
        return {
            "prompt": example["prompt_text"],
            "completion": f"{example['target_text']}{LLAMA3_EOT_TOKEN}",
        }

    return dataset.map(convert)


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


def resolve_hub_model_id(training_config: TrainingConfig) -> str | None:
    """Resolve a checkpoint repository for Hugging Face Hub pushes."""

    if not training_config.push_to_hub:
        return None
    if training_config.hub_model_id:
        return training_config.hub_model_id

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        raise ValueError(
            "--push_to_hub requires HF_TOKEN or HUGGING_FACE_HUB_TOKEN when --hub_model_id is not provided."
        )
    username = whoami(token=token)["name"]
    return f"{username}/icd10-llama3-8b-qlora-checkpoints"


def build_training_arguments(training_config : TrainingConfig, model_config: ModelConfig) -> SFTConfig:
    """Create Hugging Face training arguments for QLoRA fine-tuning."""

    ensure_dir(training_config.output_dir)
    hub_model_id = resolve_hub_model_id(training_config)
    hub_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    raw_args = {
        "output_dir": str(training_config.output_dir),
        "run_name": training_config.run_name,
        "max_length": model_config.max_seq_length,
        "max_seq_length": model_config.max_seq_length,
        "completion_only_loss": True,
        "packing": False,
        "num_train_epochs": training_config.num_train_epochs,
        "per_device_train_batch_size": training_config.per_device_train_batch_size,
        "per_device_eval_batch_size": training_config.per_device_eval_batch_size,
        "gradient_accumulation_steps": training_config.gradient_accumulation_steps,
        "learning_rate": training_config.learning_rate,
        "warmup_ratio": training_config.warmup_ratio,
        "weight_decay": training_config.weight_decay,
        "lr_scheduler_type": training_config.lr_scheduler_type,
        "optim": training_config.optim,
        "logging_steps": training_config.logging_steps,
        "eval_strategy": "steps",
        "evaluation_strategy": "steps",
        "eval_steps": training_config.eval_steps,
        "save_strategy": "steps",
        "save_steps": training_config.save_steps,
        "save_total_limit": training_config.save_total_limit,
        "max_grad_norm": training_config.max_grad_norm,
        "gradient_checkpointing": training_config.gradient_checkpointing,
        "gradient_checkpointing_kwargs": {"use_reentrant": False},
        "fp16": False,
        "bf16": False,
        "report_to": "wandb" if training_config.report_to_wandb else "none",
        "push_to_hub": training_config.push_to_hub,
        "hub_model_id": hub_model_id,
        "hub_private_repo": training_config.hub_private_repo,
        "hub_strategy": training_config.hub_strategy,
        "hub_always_push": training_config.hub_always_push,
        "hub_token": hub_token,
        "ddp_find_unused_parameters": training_config.ddp_find_unused_parameters,
        "dataloader_pin_memory": True,
        "remove_unused_columns": True,
        "logging_first_step": True,
        "group_by_length": True,
    }
    supported_args = inspect.signature(SFTConfig).parameters
    return SFTConfig(**{key: value for key, value in raw_args.items() if key in supported_args})


def build_trainer(model, tokenizer, training_args, train_dataset, eval_dataset):
    """Create SFTTrainer across TRL versions that use tokenizer or processing_class."""

    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "callbacks": [TrainableParameterCallback()],
    }
    supported_args = inspect.signature(SFTTrainer).parameters
    if "processing_class" in supported_args:
        trainer_kwargs["processing_class"] = tokenizer
    elif "tokenizer" in supported_args:
        trainer_kwargs["tokenizer"] = tokenizer
    return SFTTrainer(**trainer_kwargs)

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
    parser.add_argument("--push_to_hub", action="store_true")
    parser.add_argument("--hub_model_id", type=str, default=train_defaults.hub_model_id)
    parser.add_argument("--hub_private_repo", action=argparse.BooleanOptionalAction, default=train_defaults.hub_private_repo)
    parser.add_argument("--hub_strategy", type=str, default=train_defaults.hub_strategy)
    parser.add_argument("--hub_always_push", action="store_true")
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
        push_to_hub=args.push_to_hub,
        hub_model_id=args.hub_model_id,
        hub_private_repo=args.hub_private_repo,
        hub_strategy=args.hub_strategy,
        hub_always_push=args.hub_always_push,
        log_level=args.log_level,
    )
    seed_everything(data_config.seed)
    LOGGER.info("Hardware summary: %s", detect_compute_environment())

    dataset_dict = load_or_prepare_dataset(model_config=model_config,data_config=data_config)
    train_dataset = maybe_limit_dataset(dataset_dict["train"], training_config.max_train_samples)
    eval_dataset = maybe_limit_dataset(dataset_dict["validation"], training_config.max_eval_samples)
    train_dataset = add_prompt_completion_columns(train_dataset)
    eval_dataset = add_prompt_completion_columns(eval_dataset)
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
    if len(train_dataset) == 0:
        raise ValueError("Training dataset is empty after filtering/limiting. Check preprocessing or --max_train_samples.")
    validate_response_template(tokenizer , train_dataset[0]["text"] , model_config.response_template)
    model = load_model_for_training(model_config)
    training_args = build_training_arguments(training_config, model_config)
    trainer = build_trainer(
        model=model,
        tokenizer=tokenizer,
        training_args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
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
