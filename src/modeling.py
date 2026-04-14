"""Tokenizer, quantization, and LoRA model-loading helpers."""

from __future__ import annotations

import os

import torch
from peft import PeftModel, prepare_model_for_kbit_training, LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from src.config import ModelConfig, LLAMA3_EOT_TOKEN
from src.utils import get_logger

LOGGER = get_logger(__name__)

def get_tokenizer(model_config: ModelConfig) -> AutoTokenizer:

    tokenizer_name = model_config.tokenizer_name or model_config.model_name
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name,
        cache_dir= model_config.cache_dir,
        trust_remote_code = model_config.trust_remote_code,
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer

def build_quantization_config() -> BitsAndBytesConfig:

    return BitsAndBytesConfig(
        load_in_4bit= True,
        bnb_4bit_compute_dtype= torch.bfloat16,
        bnb_4bit_quant_type= "nf4",
        bnb_4bit_use_double_quant=True,
    )

def build_lora_config(model_config : ModelConfig) -> LoraConfig:

    return LoraConfig(
        r=model_config.lora_r,
        lora_alpha= model_config.lora_alpha,
        lora_dropout= model_config.lora_dropout,
        bias = "none",
        task_type= "CAUSAL_LM",
        target_modules= model_config.lora_target_modules,
    )

def _resolve_device_map() -> str | dict[str, int]:
    local_rank = os.environ.get("LOCAL_RANK")
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if local_rank is not None and world_size > 1:
        return {"": int(local_rank)}
    return "auto"

def load_model_for_training(model_config: ModelConfig) -> AutoModelForCausalLM:

    LOGGER.info("Loading base model %s in 4-bit mode.", model_config.model_name )
    quantization_config = build_quantization_config()
    model = AutoModelForCausalLM.from_pretrained(
        model_config.model_name,
        cache_dir=model_config.cache_dir,
        quantization_config=quantization_config,
        device_map=_resolve_device_map(),
        attn_implementation=model_config.attn_implementation,
        trust_remote_code=model_config.trust_remote_code,
    )
    model.config.use_cache = False
    if model_config.gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=model_config.gradient_checkpointing,
    )
    lora_model = get_peft_model(model, build_lora_config(model_config))
    lora_model.print_trainable_parameters()
    return lora_model

def load_model_for_inference(model_config: ModelConfig, adapter_path: str | os.PathLike[str]) -> AutoModelForCausalLM:

    LOGGER.info("Loading adapter from %s.", adapter_path)
    base_model = AutoModelForCausalLM.from_pretrained(
        model_config.model_name,
        cache_dir = model_config.cache_dir,
        trust_remote_code = model_config.trust_remote_code,
        quantization_config = build_quantization_config(),
        device_map = "auto" ,
        attn_implementation= model_config.attn_implementation
    )
    model = PeftModel.from_pretrained(base_model, adapter_path)             
    model.eval()
    return model

def get_terminal_token_ids(tokenizer: AutoTokenizer) -> list[int]:
    token_ids = [tokenizer.eos_token_id]
    eot_token_id = tokenizer.convert_tokens_to_ids(LLAMA3_EOT_TOKEN)
    if eot_token_id is not None and eot_token_id not in token_ids:
        token_ids.append(eot_token_id)
    return [token_id for token_id in token_ids if token_id is not None]