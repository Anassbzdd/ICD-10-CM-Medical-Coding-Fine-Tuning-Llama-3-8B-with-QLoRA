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
    setup_logging, to_serializable
)