"""Utility helpers for logging, filesystem I/O, reproducibility, and checkpoints."""

from __future__ import annotations

import json
import torch
import random
import logging 
import os
from dataclasses import asdict, is_dataclass
from pathlib import Path
from datetime import datetime, timezone
from typing import Any, Iterable, Optional

def setup_logging(level:str = "INFO") -> None:
    """Configure project-wide logging with a compact, timestamped format."""
    
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level= numeric_level,
        format= "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        force = True,
    )

def get_logger(name:str) -> logging.Logger:
    """Return a module logger."""

    return logging.getLogger(name)

def seed_everything(seed:int) -> None:
    """Seed Python and PyTorch for reproducible experiments."""

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def ensure_dir(path: str | Path) -> Path:
    """Create a directory if it does not exist and return the resolved path."""

    resolved = Path(path)
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved

def timestamp_utc():
    """Return an ISO-8601 UTC timestamp."""

    return datetime.now(timezone.utc).isoformat()

def _make_json_safe(value:Any) -> Any:
    if is_dataclass(value):
        return _make_json_safe(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _make_json_safe(inner) for key, inner in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_make_json_safe(item) for item in value]
    return value

def to_serializable(value:Any) -> Any:
    """Public wrapper for converting config objects into JSON-safe structures."""

    return _make_json_safe(value)

def save_json(payload:Any, path:str | Path) -> None:
    """Write JSON to disk with UTF-8 encoding and indentation."""

    destination = Path(path)
    ensure_dir(destination.parent)
    with destination.open("w", encoding="utf-8") as handle:
        json.dump(_make_json_safe(payload), handle, indent=2, ensure_ascii=False)

def save_jsonl(rows: Iterable[dict[str,Any]], path:str | Path ) -> None:
    """Write a JSONL file for predictions or reports."""

    destination = Path(path)
    ensure_dir(destination.parent)
    with destination.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(_make_json_safe(row), ensure_ascii=False) + "\n")

def latest_checkpoint(output_dir: str | Path) -> Optional[str]:
    """Return the most recent Hugging Face checkpoint directory if it exists."""

    root = Path(output_dir)
    if not root.exists():
        return None
    candidates = []
    for checkpoint_dir in root.glob("checkpoint-*"):
        step = checkpoint_dir.name.split('-')[-1]
        if step.isdigit():
            candidates.append((int(step), checkpoint_dir))
    if not candidates:
        return None
    return str(max(candidates, key=lambda item: item[0])[1])
    
def resolve_resume_checkpoint(output_dir: str | Path, resume_value: Optional[str]):
    """Resolve an explicit checkpoint path or discover the latest checkpoint."""

    if resume_value is None:
        return None
    if resume_value.lower() in {"auto", "latest"}:
        return latest_checkpoint(output_dir)
    return resume_value

def compute_percentiles(values: list[int | float], percentiles: tuple[int, ...] = (50,90,95,99)) -> dict[str,float] :
    """Compute simple percentile summaries without introducing extra dependencies."""

    if not values:
        return {}
    ordered = sorted(values)
    results: dict[str,float] = {}
    for percentile in percentiles:
        index = min(len(ordered) - 1,round((percentile/100)*(len(ordered) - 1)))
        results[f"p{percentile}"] = float(ordered[index])
    return results
    
def summarize_numeric(values: list[int|float]) -> dict[str,float]:
    """Return descriptive statistics for a numeric series."""

    if not values:
        return {}
    summary = {
        "count": float(len(values)),
        "min" : float(min(values)),
        "max" : float(max(values)),
        "mean" : float(sum(values) / len(values))
    }
    summary.update(compute_percentiles(values))
    return summary

def detect_compute_environment() -> dict[str, Any]:
    """Report GPU availability to make hardware assumptions explicit in logs."""

    gpu_names = []
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            gpu_names.append(torch.cuda.get_device_name(index))
    return {
        "cuda_available" : torch.cuda.is_available(),
        "num_gpus" : torch.cuda.device_count(),
        "gpu_names" : gpu_names,
        "bf16_supported" : bool(torch.cuda.is_available() and torch.cuda.is_bf16_supported())
    }
