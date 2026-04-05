"""ICD-10 normalization and evaluation utilities."""

from __future__ import annotations

import re
from typing import Iterable

ICD_CODE_PATTERN = re.compile(r"\b[A-TV-Z][0-9][0-9A-Z](?:\.?[0-9A-Z]{0,4})\b")


def normalize_single_code(code: str) -> str:
    """Normalize one ICD code into a consistent, dotted uppercase format."""

    cleaned = re.sub(r"[^A-Za-z0-9.]", "", code.upper())
    if not cleaned:
        return ""
    cleaned = cleaned.replace("..", ".")
    if cleaned.count(".") > 1:
        cleaned = cleaned.replace(".", "")
    if "." not in cleaned and len(cleaned) > 3:
        cleaned = f"{cleaned[:3]}.{cleaned[3:]}"
    elif "." in cleaned:
        head, tail = cleaned.split(".", 1)
        cleaned = f"{head[:3]}.{tail}"
    return cleaned.rstrip(".")


def deduplicate_preserve_order(values: Iterable[str]) -> list[str]:
    """Remove duplicates while preserving the first-seen order."""

    seen: set[str] = set()
    deduped: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        deduped.append(value)
    return deduped


def extract_codes_from_text(text: str) -> list[str]:
    """Extract code-like substrings from free-form model output."""

    if not text:
        return []
    matches = ICD_CODE_PATTERN.findall(text.upper())
    normalized = [normalize_single_code(match) for match in matches]
    return deduplicate_preserve_order(code for code in normalized if code)


def normalize_code_text(text: str) -> str:
    """Convert free-form text into a canonical comma-separated ICD code string."""

    return ", ".join(extract_codes_from_text(text))


def code_variants(code: str) -> set[str]:
    """Return dotted and undotted forms to support leakage checks."""

    normalized = normalize_single_code(code)
    if not normalized:
        return set()
    return {normalized, normalized.replace(".", "")}


def contains_code_leakage(text: str, codes: str | Iterable[str]) -> bool:
    """Detect whether a note explicitly contains any target ICD code variant."""

    if not text:
        return False
    search_space = text.upper()
    code_list = extract_codes_from_text(codes) if isinstance(codes, str) else list(codes)
    for code in code_list:
        for variant in code_variants(code):
            if variant and variant in search_space:
                return True
    return False


def compute_example_metrics(prediction: str, reference: str) -> dict[str, float]:
    """Compute set-based metrics for one prediction/reference pair."""

    pred_set = set(extract_codes_from_text(prediction))
    ref_set = set(extract_codes_from_text(reference))
    true_positive = len(pred_set & ref_set)
    precision = true_positive / len(pred_set) if pred_set else 0.0
    recall = true_positive / len(ref_set) if ref_set else 0.0
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
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
    """Aggregate example-level metrics over a full evaluation set."""

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

    per_example = [compute_example_metrics(pred, ref) for pred, ref in zip(predictions, references)]
    averaged = {
        key: sum(metric[key] for metric in per_example) / len(per_example)
        for key in per_example[0]
    }

    total_predicted = 0
    total_reference = 0
    total_true_positive = 0
    for prediction, reference in zip(predictions, references):
        pred_set = set(extract_codes_from_text(prediction))
        ref_set = set(extract_codes_from_text(reference))
        total_predicted += len(pred_set)
        total_reference += len(ref_set)
        total_true_positive += len(pred_set & ref_set)

    micro_precision = total_true_positive / total_predicted if total_predicted else 0.0
    micro_recall = total_true_positive / total_reference if total_reference else 0.0
    micro_f1 = (
        0.0
        if micro_precision + micro_recall == 0
        else 2 * micro_precision * micro_recall / (micro_precision + micro_recall)
    )

    averaged.update(
        {
            "micro_precision": micro_precision,
            "micro_recall": micro_recall,
            "micro_f1": micro_f1,
        }
    )
    return averaged
