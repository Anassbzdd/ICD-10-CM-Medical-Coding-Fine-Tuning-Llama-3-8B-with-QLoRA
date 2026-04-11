"""Dataset exploration, cleaning, leakage filtering, and split materialization."""

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
    raw_output = (example.get("output") or "").strip()
    raw_icd_code = (example.get("icd_code") or "").strip()

    if not ehr_text:
        return None , "empty_ehr_text"
    
    normalized_output = normalize_code_text(raw_output)
    normalized_icd_code = normalize_code_text(raw_icd_code)
    target_text = normalized_output or normalized_icd_code

    if not target_text:
        return None, "empty_target_text"
    
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
        "had_label_mismatch": bool(normalized_output and normalized_icd_code and normalized_icd_code != normalized_output ),
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
    return clean_dataset, dict(stats)

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
            "validation":holdout["train"],
            "test":holdout["test"],
        }
    )

def build_dataset_report(
        raw_dataset: Dataset,
        clean_dataset: Dataset,
        split_dataset_dict : DatasetDict,
        tokenizer,
        stats: dict[str, int],
        sample_size: int
) -> dict[str, Any]:
    
    effective_sample = min(sample_size, len(clean_dataset))
    sample = clean_dataset.select(range(effective_sample)) if effective_sample else clean_dataset.select([])
    prompt_token_lengths: list[int] = []
    full_token_lengths: list[int] = []
    num_codes: list[int] = []

    for row in tqdm(sample, desc="Profiling token lengths", total=effective_sample):
        prompt_token_lengths.append(len(tokenizer(row["prompt_text"],add_special_tokens=False)["input_ids"]))
        full_token_lengths.append(len(tokenizer(row["text"],add_special_tokens=False)["input_ids"]))
        num_codes.append(int(row['num_codes']))

    instructions = raw_dataset.unique("instruction")
    report = {
        "created_at_utc": timestamp_utc(),
        "raw_rows": len(raw_dataset),
        "clean_rows": len(clean_dataset),
        "splits": { split_name : len(split) for split_name , split in split_dataset_dict.items()},
        "columns": raw_dataset.column_names,
        "unique_instruction_count": len(instructions),
        "instruction_preview": instructions[:3],
        "cleaning_stats": stats,
        "sample_size_for_report": effective_sample,
        "prompt_token_length": summarize_numeric(prompt_token_lengths),
        "full_text_token_length": summarize_numeric(full_token_lengths),
        "num_codes_per_example": summarize_numeric(num_codes),
    }

    return report

def save_processed_dataset(dataset_dict: DatasetDict, processed_dir: Path) -> None:
    ensure_dir(processed_dir)
    LOGGER.info("Saving processed dataset to %s.", processed_dir)
    dataset_dict.save_to_disk(str(processed_dir))

def parse_args() -> argparse.Namespace:
    model_defaults = ModelConfig()
    data_defaults = DataConfig()

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset_name", type=str, default=data_defaults.dataset_name)
    parser.add_argument("--dataset_config_name", type=str, default=data_defaults.dataset_config_name)
    parser.add_argument("--raw_split", type=str, default=data_defaults.raw_split)
    parser.add_argument("--seed", type=int, default=data_defaults.seed)
    parser.add_argument("--train_ratio", type=float, default=data_defaults.train_ratio)
    parser.add_argument("--val_ratio", type=float, default=data_defaults.val_ratio)
    parser.add_argument("--test_ratio", type=float, default=data_defaults.test_ratio)
    parser.add_argument("--sample_report_size", type=int, default=data_defaults.sample_report_size)
    parser.add_argument("--processed_dir", type=Path, default=data_defaults.processed_dir)
    parser.add_argument("--report_path", type=Path, default=data_defaults.report_path)
    parser.add_argument("--model_name", type=str, default=model_defaults.model_name)
    parser.add_argument("--tokenizer_name", type=str, default=model_defaults.tokenizer_name)
    parser.add_argument("--cache_dir", type=str, default=model_defaults.cache_dir)
    parser.add_argument("--log_level", type=str, default="INFO")
    return parser.parse_args()

def main() -> None:
    """Load, inspect, clean, split, and save the ICD-10 dataset."""

    args = parse_args()
    setup_logging(args.log_level)

    data_config = DataConfig(
        dataset_name = args.dataset_name,
        dataset_config_name = args.dataset_config_name,
        raw_split = args.raw_split,
        seed = args.seed,
        train_ratio = args.train_ratio,
        val_ratio = args.val_ratio,
        test_ratio = args.test_ratio,
        sample_report_size= args.sample_report_size,
        processed_dir= args.processed_dir,
        report_path = args.report_path,
    )

    model_config = ModelConfig(
        model_name= args.model_name,
        tokenizer_name= args.tokenizer_name,
        cache_dir= args.cache_dir,
    )

    seed_everything(data_config.seed)
    LOGGER.info("Loading dataset %s.",data_config.dataset_name)
    raw_dataset_dict = load_dataset(
        path = data_config.dataset_name,
        name = data_config.dataset_config_name
    )
    raw_split = raw_dataset_dict[data_config.raw_split]
    LOGGER.info("Loaded raw split with %d rows and columns %s.", len(raw_split), raw_split.column_names)

    tokenizer = get_tokenizer(model_config)
    clean_dataset, stats = materialize_clean_dataset(raw_split=raw_split, tokenizer=tokenizer)
    split_dataset_dict = split_dataset(clean_dataset=clean_dataset, data_config=data_config)
    report = build_dataset_report(
        raw_dataset = raw_split,
        clean_dataset = clean_dataset,
        split_dataset_dict = split_dataset_dict,
        tokenizer = tokenizer,
        stats = stats,
        sample_size= data_config.sample_report_size ,
    )

    save_processed_dataset(split_dataset_dict, processed_dir=data_config.processed_dir)
    save_json(report, data_config.report_path)
    LOGGER.info("Saved dataset report to %s.", data_config.report_path)
    LOGGER.info("Final split sizes: %s", report["splits"])

if __name__ == "__main__":
    main()