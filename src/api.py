"""FastAPI service for serving the fine-tuned ICD-10 coding model."""

from __future__ import annotations

import argparse
import os
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI , HTTPException
from pydantic import BaseModel, Field

from src.config import ApiConfig, ModelConfig, DEFAULT_SYSTEM_PROMPT
from src.inference import ICD10Predictor
from src.utils import get_logger, setup_logging

LOGGER = get_logger(__name__)
PREDICTOR: ICD10Predictor | None = None

class PredictorRequest(BaseModel):
    """Request schema for ICD-10 prediction."""

    note: str = Field(..., description="Synthetic EHR clinical note.")
    instruction: str = Field(default=DEFAULT_SYSTEM_PROMPT, description="Optional system prompt override.")
    max_new_tokens: int = Field(default=32, ge=1, le=128)
    temperature: float = Field(default= 0.0, ge=0.0, le=2.0)
    top_p: float = Field(default=1.0 , ge=0.0, le=1.0)
    do_sample: bool = Field(default = False)
    num_beams: int = Field(default=1, ge=1, le=8)

class PredictResponse(BaseModel):
    """Response schema for ICD-10 prediction."""

    raw_prediction:str
    normalized_prediction:str
    codes: list[str]

def build_model_config_from_env() -> tuple[ModelConfig, Path]:

    adapter_path = os.environ.get("ADAPTER_PATH")
    if adapter_path is None:
        raise ValueError("ADAPTER_PATH environment variable is required.")
    model_config = ModelConfig(
        model_name= os.environ.get("MODEL_NAME",ModelConfig().model_name),
        tokenizer_name=os.environ.get("TOKENIZER_NAME",ModelConfig().tokenizer_name),
        cache_dir=os.environ.get("CACHE_DIR",ModelConfig().cache_dir),
        max_seq_length=int(os.environ.get("MAX_SEQ_LENGTH",ModelConfig().max_seq_length)),
    )
    return model_config, Path(adapter_path)

@asynccontextmanager
async def lifespan(_:FastAPI):

    global PREDICTOR
    model_config, adapter_path = build_model_config_from_env()
    PREDICTOR = ICD10Predictor(model_config,adapter_path)
    try:
        yield
    finally:
        PREDICTOR = None

app = FastAPI(title= "ICD-10 QLoRA API", version="1.0.0", lifespan= lifespan)

@app.get("/health")
def health() -> dict[str,str]:
    return {"status":"ok"}

@app.post("/predict", response_model= PredictResponse)
def predict(request: PredictorRequest) -> PredictResponse:
    if PREDICTOR is None:
        raise HTTPException(status_code=503, detail="Model is still loading.")
    prediction = PREDICTOR.predict(
        clinical_note = request.note,
        instruction = request.instruction,
        max_new_tokens = request.max_new_tokens,
        temperature = request.temperature,
        top_p = request.top_p,
        do_sample = request.do_sample,
        num_beams = request.num_beams,
    )
    return PredictResponse(
        raw_prediction = str(prediction["raw_prediction"]),
        normalized_prediction = str(prediction["normalized_prediction"]),
        codes = list(prediction["codes"]),
    )

def parse_args() -> argparse.Namespace:

    api_defaults = ApiConfig()
    model_defaults = ModelConfig()

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter_path", type=Path, required=True)
    parser.add_argument("--model_name", type=str, default=model_defaults.model_name)
    parser.add_argument("--tokenizer_name", type=str, default=model_defaults.tokenizer_name)
    parser.add_argument("--cache_dir", type=str, default=model_defaults.cache_dir)
    parser.add_argument("--max_seq_length", type=int, default=model_defaults.max_seq_length)
    parser.add_argument("--host", type=str, default=api_defaults.host)
    parser.add_argument("--port", type=int, default=api_defaults.port)
    parser.add_argument("--workers", type=int, default=api_defaults.workers)
    parser.add_argument("--log_level", type=str, default=api_defaults.log_level)
    return parser.parse_args()

def main() -> None:

    args = parse_args()
    setup_logging(args.log_level)
    os.environ["ADAPTER_PATH"] = str(args.adapter_path)
    os.environ["MODEL_NAME"] = args.model_name
    if args.cache_dir is not None:
        os.environ["CACHE_DIR"] = args.cache_dir
    if args.tokenizer_name is not None:
        os.environ["TOKENIZER_NAME"] = args.tokenizer_name
    os.environ["MAX_SEQ_LENGTH"] = str(args.max_seq_length)

    uvicorn.run(
        "src.api:app",
        host= args.host,
        port= args.port,
        workers= args.workers,
        log_level= args.log_level,
    )

if __name__ == "__main__":
    main()
