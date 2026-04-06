from __future__ import annotations

import argparse
import hashlib
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from datasets import Dataset, Value, load_dataset, DatasetDict, Features
from tqdm.auto import tqdm

from src.config import DataConfig, ModelConfig
from src.metrics import contains_code_leakage, extract_codes_from_text, normalize_code_text
from src.modeling import get_tokenizer
from src.prompting import build_prompt_text, build_training_text
from src.utils import (
    ensure_dir,
    get_logger,
    save_json,
    seed_everything,
    setup_logging,
    summarize_numeric,
    timestamp_utc,
)
