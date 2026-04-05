from __future__ import annotations

import re
from typing import Iterable

ICD_CODE_PATTERN = re.compile(r"\b[A-TV-Z][0-9][0-9A-Z](?:\.?[0-9A-Z]{0,4})\b")

def normalize_single_code(code:str) -> str:
    """Clean and format a single ICD code."""

    cleaned = re.sub(f"[^A-Za-z0-9.]", "", code.upper())
    if not cleaned:
        return ""
    cleaned = cleaned.replace("..", ".")
    if cleaned.count(".") > 1:
        cleaned = cleaned.replace(".", "")
    if "." not in cleaned and len(cleaned) > 3:
        cleaned = f"{cleaned[:3]}.{cleaned[3:]}"
    elif "." in cleaned:
        head, tail = cleaned.split('.')
        cleaned = f"{head[:3]}.{tail}"
    return cleaned.rstrip(".")

def deduplicate_perserve_order(values: Iterable[str]) -> list[str]:
    """Remove duplicates, keep original order"""

    seen: set[str] = set()
    deduped: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        deduped.append(value)
    return deduped

def extract_codes_from_text(text: str) -> list[str]:
    """Extract code from the text"""

    if not text:
        return []
    matches = ICD_CODE_PATTERN.findall(text.upper())
    normalized = [normalize_single_code(match) for match in matches]
    return deduplicate_perserve_order( code for code in normalized if code)

def normalize_code_text(text:str) -> str : 
    """Convert text into a canonical comma-separated """

    return ", ".join(extract_codes_from_text(text))

def code_varients(code: str) -> set[str]:
    """Return dotted and undotted forms"""

    normalized = normalize_single_code(code)
    if not normalized:
        return set()
    return {normalized, normalized.replace(".","")}

def contains_code_leakage(text:str, code: str | Iterable[str]):
    if not text:
        False
    search_space = text.upper()
    code_list = extract_codes_from_text(code) if isinstance(code, str) else list(code)
    for code in code_list:
        for varient in code_varients(code):
            if varient and varient in search_space:
                return True
    return False

def compute_example_metrics(prediction:str, reference:str) -> dict[str,float]:
    pred_set = set(extract_codes_from_text(prediction))
    ref_set = set(extract_codes_from_text(reference))
    true_positive = len(pred_set & ref_set)
    precision = true_positive / len(pred_set) if pred_set else 0.0
    recall = true_positive / len(pred_set) if ref_set else 0.0
    f1 = 0 if precision + recall == 0 else 2*(precision*recall) / precision + recall
    exact_match = 1.0 if pred_set == ref_set else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "exact_match": exact_match,
        "num_predicted_codes": float(len(pred_set)),
        "num_reference_codes": float(len(ref_set)),
    }

def aggregate_metrics(predictions: list[str], references: list[str]) -> dict[str, float]:
    if len(predictions) != len(references):
        raise ValueError("Predictions and references must have the same length.")
    if not predictions:
        return {
            "exact_match": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "micro_precision": 0.0,
            "micro_recall": 0.0,
            "micro_f1": 0.0,
        }
    per_example = [ compute_example_metrics(pred, ref) for pred , ref in zip(predictions, references)]
    averaged = {
        key: sum(metric[key] for metric in per_example) / len(per_example)
        for key in per_example[0]
    }

    total_predicted = 0
    total_reference = 0
    total_true_positive = 0
    for predictions, reference in zip(predictions, references):
        pred_set = set(extract_codes_from_text(predictions))
        ref_set = set(extract_codes_from_text(references))
        total_predicted += len(pred_set)
        total_reference += len(ref_set)
        total_true_positive += len(pred_set & ref_set)

    micro_precision = total_true_positive / total_predicted if total_predicted else 0.0
    micro_recall = total_true_positive / total_reference if total_predicted else 0.0
    micro_f1 = (
        0.0
        if micro_precision + micro_recall == 0 else 
        2*(micro_precision * micro_recall) / micro_precision + micro_recall
    )

    averaged.update(
        {
            "micro_precision": micro_precision,
            "micro_recall": micro_recall,
            "micro_f1": micro_f1,
        }
    )
    return averaged