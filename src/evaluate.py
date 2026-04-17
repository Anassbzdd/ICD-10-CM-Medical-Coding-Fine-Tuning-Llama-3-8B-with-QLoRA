"""Offline generation-based evaluation for the fine-tuned ICD-10 model."""

from __future__ import annotations

import argparse
from pathlib import Path

import wandb
import torch
from datasets import load_from_disk
from tqdm.auto import tqdm

from src.config import DataConfig, ModelConfig, EvaluationModel
from src.metrics import aggregate_metrics
from src.modeling import get_terminator_token_ids, get_tokenizer, load_model_for_inference
from src.utils import ensure_dir, get_logger, save_json, save_jsonl, setup_logging

LOGGER = get_logger(__name__)

