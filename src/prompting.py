"""Prompt builders for Llama 3 ICD-10 completion training and inference."""

from __future__ import annotations

from src.config import DEFAULT_SYSTEM_PROMPT, LLAMA3_EOT_TOKEN
from src.metrics import normalize_code_text

BOS_TOKEN = "<|begin_of_text|>"
START_HEADER = "<|start_header_id|>"
END_HEADER = "<|end_header_id|>"


def _header(role: str) -> str:
    """Build a Llama 3 chat header without relying on tokenizer.chat_template."""

    return f"{START_HEADER}{role}{END_HEADER}\n\n"


def build_messages(instruction: str, clinical_note: str) -> list[dict[str, str]]:
    """Return semantic chat messages for debugging and future instruct models."""

    system_prompt = instruction.strip() if instruction and instruction.strip() else DEFAULT_SYSTEM_PROMPT
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": clinical_note.strip()},
    ]


def build_prompt_text(tokenizer, instruction: str, clinical_note: str) -> str:
    """Build the generation prompt manually for base Llama 3 tokenizers."""

    del tokenizer
    system_prompt = instruction.strip() if instruction and instruction.strip() else DEFAULT_SYSTEM_PROMPT
    note = clinical_note.strip()
    return (
        f"{BOS_TOKEN}"
        f"{_header('system')}{system_prompt}{LLAMA3_EOT_TOKEN}"
        f"{_header('user')}{note}{LLAMA3_EOT_TOKEN}"
        f"{_header('assistant')}"
    )


def build_training_text(prompt_text: str, target_codes: str) -> str:
    """Append the normalized ICD completion and end-of-turn token."""

    normalized_target = normalize_code_text(target_codes)
    if not normalized_target:
        raise ValueError("Target ICD code text is empty after normalization.")
    return f"{prompt_text}{normalized_target}{LLAMA3_EOT_TOKEN}"