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

    if data_config.processed_dir.exist():
        LOGGER.info("Loading processed dataset from %s.", data_config.processed_dir)
        return load_from_disk(str(data_config.processed_dir))
    
    LOGGER.warning(
        "Processed dataset not found at %s. Building it inline so training can proceed.",
        data_config.processed_dir,
    )
    tokenizer = get_tokenizer(model_config.tokenizer_name)
    raw_dataset_dict = load_dataset(data_config.dataset_name, data_config.dataset_config_name)
    raw_split = raw_dataset_dict[data_config.raw_split]
    clean_dataset, stats = materialize_clean_dataset(raw_split, tokenizer)
    split_dataset_dict = split_dataset(clean_dataset,tokenizer)
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
    return dataset.range(min(len(dataset), max_samples))

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

def build_training_arguments(training_config : TrainingConfig) -> TrainingConfig:
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

    model_config = ModelConfig()
    data_config = DataConfig()
    training_config = TrainingConfig()

    parser = argparse.ArgumentParser(description=__doc__)
    