"""Shared configuration objects and defaults for the ICD-10 QLoRA project."""

from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_ROOT = PROJECT_ROOT / "artifacts"
DATA_ROOT = PROJECT_ROOT / "data"
REPORTS_ROOT = PROJECT_ROOT / "reports"
CHECKPOINT_ROOT = PROJECT_ROOT / "checkpoints"
PREDICTIONS_ROOT = PROJECT_ROOT / "predictions"
LOGS_ROOT = PROJECT_ROOT / "logs"

DEFAULT_MODEL_NAME = "meta-llama/Meta-Llama-3-8B"
DEFAULT_DATASET_NAME = "generative-technologies/synth-ehr-icd10-llama3-format"
DEFAULT_SYSTEM_PROMPT = (
    "You are a medical coding assistant. Your task is to analyze the given "
    "electronic health record, and provide a list of appropriate ICD-10-CM codes "
    "based on the details mentioned in the note. If multiple codes are applicable, "
    "separate them with commas. Respond with the ICD-10-CM codes only, without any "
    "additional explanations or context."
)
LLAMA3_ASSISTANT_HEADER = "<|start_header_id|>assistant<|end_header_id|>\n\n"
LLAMA3_EOT_TOKEN = "<|eot_id|>"
DEFAULT_LORA_TARGET_MODULES = ("q_proj","k_proj","v_proj","o_proj")

@dataclass
class ModelConfig:
    model_name:str = DEFAULT_MODEL_NAME
    tokenizer_name:Optional[str] = None
    cache_dir: Optional[str] = None
    max_seq_length: int = 1024
    attn_implementation:str = "sdpa"
    load_in_4bit: bool = True
    bnb_4bit_quant_type:str = "nf4"
    bnb_4bit_use_double_quant: bool = True
    lora_r:int = 16
    lora_alpha:int = 32
    lora_dropout:float = 0.05
    lora_target_modules: Tuple = DEFAULT_LORA_TARGET_MODULES
    gradient_checkpointing: bool = True
    response_template: str = LLAMA3_ASSISTANT_HEADER
    trust_remote_code:bool = False

@dataclass
class DataConfig:
    dataset_name:str = DEFAULT_DATASET_NAME
    dataset_config_name:Optional[str] = None
    raw_split:str = "train"
    seed:int = 42
    train_ratio:float = 0.9
    val_ratio:float = 0.05
    test_ratio:float = 0.05
    num_proc: Optional[int] = None
    sample_report_size:int = 1000
    processed_dir: Path = field(default_factory=lambda: DATA_ROOT / "processed")
    report_path: Path = field(default_factory=lambda: REPORTS_ROOT / "dataset_report.json" )
    save_name: str = "synth_ehr_icd10_llama3_clean"

@dataclass
class TrainingConfig:
    output_dir: Path = field(default_factory=lambda: CHECKPOINT_ROOT/ "llama3-8b-qlora-icd10")
    run_name:str = "llama3-8b-qlora-icd10"
    per_device_train_batch_size:int = 1
    per_device_eval_batch_size:int = 1
    gradient_accumulation_steps: int = 16
    num_train_epochs:float = 1.0
    learning_rate:float = 2e-4
    warmup_ratio: float = 0.03
    weight_decay: float = 0.01
    lr_scheduler_type:str = "cosine"
    optim:str = "paged_adamw_8bit"
    logging_steps:int = 10
    eval_steps:int = 200
    save_steps:int = 200
    save_total_limit:int = 3
    wandb_entity:Optional[str] = None
    max_grad_norm:float = 0.3
    gradient_checkpointing: bool = True
    log_level:str = "INFO"
    wandb_project:str = "medical-coding-qlora"
    report_to_wandb:bool = True
    max_train_samples:Optional[int] = None
    max_eval_samples:Optional[int] = 2048
    ddp_find_unused_parameters:bool = False
    resume_from_checkpoint:Optional[str] = None



@dataclass
class EvaluationModel:
    split: str = "test"
    batch_size: int = 4
    max_new_tokens: int = 32
    temperature: float = 0.0
    top_p: float = 1.0
    do_sample: bool = False
    num_beams: int = 1
    max_samples: Optional[int] = 1024
    output_path: Path = field(default_factory=lambda: PREDICTIONS_ROOT / "test_predictions.jsonl")
    wandb_project: str = "medical-coding-qlora"
    wandb_entity: Optional[str] = None
    run_name: str = "llama3-8b-qlora-icd10-eval"
    log_level: str = "INFO"
    

@dataclass
class InferenceConfig:
    max_new_tokens:int = 32
    temperature:float = 0.0
    do_sample:bool = False
    top_p:float = 1.0
    num_beams:int = 1
    log_level:str = "INFO"

@dataclass
class ApiConfig:
    host:str = "0.0.0.0"
    port:int = 8000
    workers:int = 1
    log_level:str = "INFO"
