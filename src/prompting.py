from __future__ import annotations

from transformers import PreTrainedTokenizerBase
from src.config import DEFAULT_SYSTEM_PROMPT, LLAMA3_EOT_TOKEN

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