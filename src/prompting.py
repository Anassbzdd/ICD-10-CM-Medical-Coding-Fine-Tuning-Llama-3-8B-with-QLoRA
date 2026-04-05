from __future__ import annotations

from transformers import PreTrainedTokenizerBase
from src.config import DEFAULT_SYSTEM_PROMPT, LLAMA3_EOT_TOKEN
from src.metrics import normalize_code_text

def build_messages(instruction:str, clinical_note:str) -> list[dict[str,str]]:
    system_prompt = instruction.strip() if instruction and instruction.strip() else DEFAULT_SYSTEM_PROMPT
    return [
        {"role": "system" , "content": system_prompt},
        {"role": "user" , "content": clinical_note.strip()}
    ]

def build_prompt_text(
        tokenizer:PreTrainedTokenizerBase,
        instruction:str, 
        clinical_note:str,
) -> str :
    return tokenizer.apply_chat_template(
        build_messages(instruction=instruction, clinical_note=clinical_note),
        tokenize=False,
        add_generation_prompt=True,
    )

def build_training_text(prompt_text: str, target_codes:str) -> str:
    normalized_target = normalize_code_text(target_codes)
    if not normalized_target:
        raise ValueError("Target ICD code text is empty after normalization.")
    if prompt_text.endswith(LLAMA3_EOT_TOKEN):
        return prompt_text + normalized_target
    return f"{prompt_text}{normalized_target}{LLAMA3_EOT_TOKEN}"