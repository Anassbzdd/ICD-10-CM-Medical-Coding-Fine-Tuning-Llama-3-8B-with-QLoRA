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
    return cleaned , None

def materialize_clean_dataset(raw_split: Dataset, tokenizer) -> tuple[Dataset, dict[str,int]]: 
    seen_ids : set[str] = set()
    stats: Counter[str] = Counter()

    def generator() -> Iterable[dict[str, Any]]:
        for example in tqdm(raw_split, desc="Cleaning dataset", total=len(raw_split)):
            stats['rows_seen'] += 1
            cleaned , skip_reason = clean_example(example, tokenizer)
            if cleaned is None:
                stats[skip_reason or "unknown_skip_reason"] += 1
                continue
            if cleaned['example_id'] in seen_ids :
                stats['duplicate_rows'] += 1
                continue
            seen_ids.add(cleaned["example_id"])
            stats["rows_kept"] += 1
            if cleaned["had_label_mismatch"]:
                stats["label_mismatches"] += 1
            yield cleaned

    clean_dataset = Dataset.from_generator(generator, features=CLEAN_FEATURES)
    return clean_dataset, stats

def split_dataset(clean_dataset: Dataset, data_config: DataConfig) -> DatasetDict:

    if round(data_config.train_ratio + data_config.test_ratio + data_config.val_ratio, 6) != 1.0:
        raise ValueError("Train/val/test ratios must sum to 1.0.") 
    
    LOGGER.info(
        "Splitting cleaned dataset with ratios train=%.3f val=%.3f test=%.3f.",
        data_config.train_ratio,
        data_config.test_ratio,
        data_config.val_ratio
    )

    initial = clean_dataset.train_test_split(
        test_size = data_config.test_ratio + data_config.val_ratio,
        seed = data_config.seed,
        shuffle = True,
    )

    holdout = initial["test"].train_test_split(
        test_size= data_config.test_ratio / (data_config.test_ratio + data_config.val_ratio),
        seed = data_config.seed,
        shuffle = True,
    )
    return DatasetDict(
        {
            "train":initial["train"],
            "test":holdout["train"],
            "validation":holdout["test"],
        }
    )

def build_dataset_report(
        raw_dataset: Dataset,
        clean_dataset: Dataset,
        split_data_dict : DatasetDict,
        tokenizer,
        stats: dict[str, int],
        sample_size: int
) -> dict[str, Any]:
