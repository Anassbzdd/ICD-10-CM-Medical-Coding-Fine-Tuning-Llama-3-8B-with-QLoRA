"""Reusable predictor and CLI inference entrypoint for ICD-10 generation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from tqdm.auto import tqdm

from src.config import DEFAULT_SYSTEM_PROMPT, InferenceConfig, ModelConfig
from src.metrics import extract_codes_from_text, normalize_code_text
from src.modeling import get_terminator_token_ids, get_tokenizer, load_model_for_inference
from src.prompting import build_prompt_text
from src.utils import get_logger, save_jsonl, setup_logging

LOGGER = get_logger(__name__)

class ICD10Predictor:
    """Thin inference wrapper that keeps prompt building and decoding consistent."""

    def __init__(self,model_config:ModelConfig ,adapter_path: str | Path) -> None:
        self.model_config = model_config
        self.tokenizer = get_tokenizer(model_config)
        self.model = load_model_for_inference(model_config, adapter_path)
    
    def predict(
            self,
            clinical_note:str,
            instruction:str = DEFAULT_SYSTEM_PROMPT,
            max_new_tokens:int = 32,
            temperature:float = 0.0,
            top_p:float = 1.0,
            do_sample:bool = False,
            num_beams:int = 1
    ) -> dict[str, object]:
        """Generate ICD-10 codes for one clinical note."""

        prompt_text = build_prompt_text(
            tokenizer=self.tokenizer,
            instruction=instruction,
            clinical_note=clinical_note,
        )
        encoded = self.tokenizer(
            prompt_text,
            return_tensors="pt",
            truncation=True,
            max_length = self.model_config.max_seq_length,
        ).to(self.model.device)
        with torch.inference_mode():
            generated = self.model.generate(
                **encoded,
                do_sample= do_sample,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                num_beams=num_beams,
                eos_token_id = get_terminator_token_ids(self.tokenizer),
                pad_token_id = self.tokenizer.pad_token_id,
            )
        completion_tokens = generated[0][encoded["input_ids"].shape[1]:]
        raw_prediction = self.tokenizer.decode(
            completion_tokens,
            skip_special_tokens = True,
        ).strip()
        normalized_prediction = normalize_code_text(raw_prediction)
        return {
            "prompt_text": prompt_text,
            "raw_prediction": raw_prediction,
            "normalized_prediction": normalized_prediction,
            "codes": extract_codes_from_text(raw_prediction)
        }

def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for local inference."""

    model_defaults = ModelConfig()
    inference_defaults = InferenceConfig()

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter_path", type=Path, required=True)
    parser.add_argument("--model_name", type=str, default=model_defaults.model_name)
    parser.add_argument("--tokenizer_name", type=str, default=model_defaults.tokenizer_name)
    parser.add_argument("--cache_dir", type=str, default=model_defaults.cache_dir)
    parser.add_argument("--max_seq_length", type=int, default=model_defaults.max_seq_length)
    parser.add_argument("--note", type=str, default=None)
    parser.add_argument("--note_file", type=Path, default=None)
    parser.add_argument("--input_jsonl", type=Path, default=None)
    parser.add_argument("--output_jsonl", type=Path, default=None)
    parser.add_argument("--instruction", type=str, default=DEFAULT_SYSTEM_PROMPT)
    parser.add_argument("--max_new_tokens", type=int, default=inference_defaults.max_new_tokens)
    parser.add_argument("--temperature", type=float, default=inference_defaults.temperature)
    parser.add_argument("--top_p", type=float, default=inference_defaults.top_p)
    parser.add_argument("--do_sample", action="store_true")
    parser.add_argument("--num_beams", type=int, default=inference_defaults.num_beams)
    parser.add_argument("--log_level", type=str, default=inference_defaults.log_level)
    return parser.parse_args()

def load_notes_from_jsonl(path: Path) -> list[dict[str, str]]:

    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows

def main() -> None:
    """Run one-off or batch inference from the command line."""

    args = parse_args()
    setup_logging(args.log_level)

    model_config = ModelConfig(
        model_name=args.model_name,
        tokenizer_name=args.tokenizer_name,
        cache_dir=args.cache_dir,
        max_seq_length=args.max_seq_length,
    )
    predictor = ICD10Predictor(model_config= model_config,adapter_path = args.adapter_path)

    if args.input_jsonl is not None:
        rows = load_notes_from_jsonl(args.input_jsonl)
        outputs = []
        for row in tqdm(rows, desc="Running batch inference"):
            prediction = predictor.predict(
                clinical_note= row["note"],
                instruction= row.get("instruction",args.instruction),
                max_new_tokens= args.max_new_tokens,
                temperature= args.temperature,
                top_p= args.top_p,
                do_sample= args.do_sample,
                num_beams= args.num_beams
            )
            outputs.append({
                "note": row["note"],
                **prediction,
            })
        if args.output_jsonl is None:
            raise ValueError("--output_jsonl is required when --input_jsonl is provided.")
        save_jsonl(outputs, args.output_jsonl)
        LOGGER.info("Saved %d predictions to %s.", len(outputs), args.output_jsonl)
        return
        
    if args.note_file is not None:
        clinical_note = args.note_file.read_text(encoding= "utf-8")
    elif args.note is not None:
        clinical_note = args.note
    else:
        raise ValueError("Provide either --note, --note_file, or --input_jsonl.")
        
    prediction = predictor.predict(
            clinical_note= clinical_note,
            instruction= args.instruction,
            max_new_tokens= args.max_new_tokens,
            temperature= args.temperature,
            top_p= args.top_p,
            do_sample= args.do_sample,
            num_beams= args.num_beams
    )
    print(json.dumps(prediction, indent=2))
        

if __name__ == "__main__":
    main()

