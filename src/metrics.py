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

