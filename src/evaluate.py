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

def batched_generate(
        model,
        tokenizer,
        prompts: list[str],
        max_seq_length: int,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        do_sample: bool,
        num_beams:int,
) -> list[str] :
    """Generate ICD code predictions for a batch of prompt strings."""

    encoded = tokenizer(
        prompts,
        return_tensors="pt",
        padding = True,
        truncation = True,
        max_length = max_seq_length,
    )
    encoded = {key : value.to(model.device)for key , value in encoded.items()}
    with torch.inference_mode():
        generated = model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            num_beams=num_beams,
            eos_token_id = get_terminator_token_ids(tokenizer),
            pad_token_id = tokenizer.pad_token_id,
        )
    
    predictions: list[str] = []
    input_lenght = encoded["attention_mask"].sum(dim=1).tolist()
    for sequence, prompt_lenght in zip(generated, input_lenght):
        completion_task = sequence[int(prompt_lenght):]
        predictions.append(tokenizer.decode(completion_task ,skip_special_tokens = True).strip())
    return predictions

def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for offline evaluation."""

    data_defaults = DataConfig()
    model_defaults = ModelConfig()
    eval_defaults = EvaluationModel()

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed_dir", type=Path, default=data_defaults.processed_dir)
    parser.add_argument("--adapter_path", type=Path, required=True)
    parser.add_argument("--model_name", type=str, default=model_defaults.model_name)
    parser.add_argument("--tokenizer_name", type=str, default=model_defaults.tokenizer_name)
    parser.add_argument("--cache_dir", type=str, default=model_defaults.cache_dir)
    parser.add_argument("--max_seq_length", type=int, default=model_defaults.max_seq_length)
    parser.add_argument("--split", type=str, choices=["validation", "test", "train"], default=eval_defaults.split)
    parser.add_argument("--batch_size", type=int, default=eval_defaults.batch_size)
    parser.add_argument("--max_new_tokens", type=int, default=eval_defaults.max_new_tokens)
    parser.add_argument("--temperature", type=float, default=eval_defaults.temperature)
    parser.add_argument("--top_p", type=float, default=eval_defaults.top_p)
    parser.add_argument("--do_sample", action="store_true")
    parser.add_argument("--num_beams", type=int, default=eval_defaults.num_beams)
    parser.add_argument("--max_samples", type=int, default=eval_defaults.max_samples)
    parser.add_argument("--output_path", type=Path, default=eval_defaults.output_path)
    parser.add_argument("--wandb_project", type=str, default=eval_defaults.wandb_project)
    parser.add_argument("--wandb_entity", type=str, default=eval_defaults.wandb_entity)
    parser.add_argument("--run_name", type=str, default=eval_defaults.run_name)
    parser.add_argument("--no_wandb", action="store_true")
    parser.add_argument("--log_level", type=str, default=eval_defaults.log_level)
    return parser.parse_args()

def main() -> None:
    """Load a saved adapter and evaluate it on a held-out dataset split."""

    args = parse_args()
    setup_logging(args.log_level)