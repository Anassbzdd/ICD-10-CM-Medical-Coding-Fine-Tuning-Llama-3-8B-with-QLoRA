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

LOGGER = get_logger(__name__)

CLEAN_FEATURES = Features(
    {
        "example_id": Value("string"),
        "ehr_text": Value("string"),
        "instruction": Value("string"),
        "raw_icd_code": Value("string"),
        "raw_output": Value("string"),
        "target_text": Value("string"),
        "prompt_text": Value("string"),
        "text": Value("string"),
        "num_codes": Value("int32"),
        "had_label_mismatch": Value("bool"),
    }
)

def compute_example_id(clinical_note:str , target_text:str) -> str:
    payload = f"{clinical_note.strip()}|||{target_text.strip()}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()

def clean_example(example: dict[str, Any], tokenizer) -> tuple[dict[str, Any] | None , str | None]:
    ehr_text = (example.get("ehr_text") or "").strip()
    instruction = (example.get("instruction") or "").strip()
    raw_output = (example.get("raw_output") or "").strip()
    raw_icd_code = (example.get("raw_icd_code") or "").strip()

    if not ehr_text:
        return None , "empty_ehr_text"
    
    normalized_output = normalize_code_text(raw_output)
    normalized_icd_code = normalize_code_text(raw_icd_code)
    target_text = normalized_output or normalized_icd_code

    if contains_code_leakage(ehr_text,target_text):
        return None , "label_leakage_in_ehr_text"
    
    prompt_text = build_prompt_text(tokenizer=tokenizer, instruction=instruction, clinical_note=ehr_text)
    if contains_code_leakage(prompt_text,target_text):
        return None , "label_leakage_in_prompt"
    
    try:
        training_text = build_training_text(prompt_text= prompt_text, target_codes= target_text)
    except ValueError:
        return None, "invalid_training_text"
    
    cleaned = {
        "example_id": compute_example_id(ehr_text, target_text),
        "ehr_text": ehr_text,
        "instruction": instruction,
        "raw_icd_code": raw_icd_code,
        "raw_output": raw_output,
        "target_text": target_text,
        "prompt_text": prompt_text,
        "text": training_text,
        "num_codes": len(extract_codes_from_text(target_text)),
        "had_label_mismatch": bool(normalized_output,normalized_icd_code, normalized_icd_code != normalized_output ),
    }