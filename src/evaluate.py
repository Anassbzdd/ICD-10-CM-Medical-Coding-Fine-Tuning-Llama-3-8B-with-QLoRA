"""Offline generation-based evaluation for the fine-tuned ICD-10 model."""

from __future__ import annotations

import argparse
from pathlib import Path

import wandb
import torch
from datasets import load_from_disk
from tqdm.auto import tqdm

from src.config import DataConfig, ModelConfig, EvaluationModel
from src.metrics import aggregate_metrics
from src.modeling import get_terminator_token_ids, get_tokenizer, load_model_for_inference
from src.utils import ensure_dir, get_logger, save_json, save_jsonl, setup_logging

LOGGER = get_logger(__name__)

def batched_generate(
        model,
        tokenizer,
        prompts: list[str],
        max_seq_length: int,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        do_sample: bool,
        num_beams:int,
) -> list[str] :
    """Generate ICD code predictions for a batch of prompt strings."""

    encoded = tokenizer(
        prompts,
        return_tensors="pt",
        padding = True,
        truncation = True,
        max_length = max_seq_length,
    )
    encoded = {key : value.to(model.device)for key , value in encoded.items()}
    with torch.inference_mode():
        generated = model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            num_beams=num_beams,
            eos_token_id = get_terminator_token_ids(tokenizer),
            pad_token_id = tokenizer.pad_token_id,
        )
    
    predictions: list[str] = []
    input_lenght = encoded["attention_mask"].sum(dim=1).tolist()
    for sequence, prompt_lenght in zip(generated, input_lenght):
        completion_task = sequence[int(prompt_lenght):]
        predictions.append(tokenizer.decode(completion_task ,skip_special_tokens = True).strip())
    return predictions

def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for offline evaluation."""

    data_defaults = DataConfig()
    model_defaults = ModelConfig()
    eval_defaults = EvaluationModel()

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed_dir", type=Path, default=data_defaults.processed_dir)
    parser.add_argument("--adapter_path", type=Path, required=True)
    parser.add_argument("--model_name", type=str, default=model_defaults.model_name)
    parser.add_argument("--tokenizer_name", type=str, default=model_defaults.tokenizer_name)
    parser.add_argument("--cache_dir", type=str, default=model_defaults.cache_dir)
    parser.add_argument("--max_seq_length", type=int, default=model_defaults.max_seq_length)
    parser.add_argument("--split", type=str, choices=["validation", "test", "train"], default=eval_defaults.split)
    parser.add_argument("--batch_size", type=int, default=eval_defaults.batch_size)
    parser.add_argument("--max_new_tokens", type=int, default=eval_defaults.max_new_tokens)
    parser.add_argument("--temperature", type=float, default=eval_defaults.temperature)
    parser.add_argument("--top_p", type=float, default=eval_defaults.top_p)
    parser.add_argument("--do_sample", action="store_true")
    parser.add_argument("--num_beams", type=int, default=eval_defaults.num_beams)
    parser.add_argument("--max_samples", type=int, default=eval_defaults.max_samples)
    parser.add_argument("--output_path", type=Path, default=eval_defaults.output_path)
    parser.add_argument("--wandb_project", type=str, default=eval_defaults.wandb_project)
    parser.add_argument("--wandb_entity", type=str, default=eval_defaults.wandb_entity)
    parser.add_argument("--run_name", type=str, default=eval_defaults.run_name)
    parser.add_argument("--no_wandb", action="store_true")
    parser.add_argument("--log_level", type=str, default=eval_defaults.log_level)
    return parser.parse_args()

def main() -> None:
    """Load a saved adapter and evaluate it on a held-out dataset split."""

    args = parse_args()
    setup_logging(args.log_level)

    model_config = ModelConfig(
        model_name=args.model_name,
        tokenizer_name=args.tokenizer_name,
        cache_dir=args.cache_dir,
        max_seq_length=args.max_seq_length,
    )

    tokenizer = get_tokenizer(model_config)
    model = load_model_for_inference(model_config, args.adapter_path)

    dataset_dict = load_from_disk(str(args.processed_dir))
    dataset = dataset_dict[args.split]
    if args.max_samples is not None:
        dataset = dataset.select(range(min(len(dataset), args.max_samples)))
    LOGGER.info("Evaluating %d rows from split '%s'.",len(dataset), args.split)

    if not args.no_wandb:
        wandb.init(
            project= args.wandb_project,
            entity = args.wandb_entity,
            name = args.run_name,
            job_type="evaluation",
            config={
                "split": args.split,
                "batch_size": args.batch_size,
                "max_new_tokens": args.max_new_tokens,
                "adapter_path": str(args.adapter_path),
            }
        )
    
    prompts = dataset["prompt_text"]
    references = dataset["target_text"]
    predictions: list[str] = []

    for start_index in tqdm(range(0,len(dataset),args.batch_size), desc="Generating predictions"):
        stop_index = min(start_index + args.batch_size , len(dataset))
        batch_prompts = prompts[start_index:stop_index]
        batch_predictions = batched_generate(
            model=model,
            tokenizer=tokenizer,
            prompts= batch_prompts,
            max_seq_length=model_config.max_seq_length,
            max_new_tokens= args.max_new_tokens,
            temperature=args.temperature,
            top_p = args.top_p,
            do_sample=args.do_sample,
            num_beams=args.num_beams,
        )
        predictions.extend(batch_predictions)
    
    metrics = aggregate_metrics(predictions=predictions, references=references)
    LOGGER.info("Evaluation metrics: %s", metrics)

    prediction_rows = [
        {
            "example_id":dataset[index]["example_id"],
            "prompt_text":dataset[index]["prompt_text"],
            "reference": references[index],
            "prediction": predictions[index],
        }
        for index in range(len(dataset))
    ]

    ensure_dir(args.output_path.parent)
    save_jsonl(prediction_rows, args.output_path)
    save_json(
        {
            "split":args.split,
            "metrics":metrics,
            "adapter_path":str(args.adapter_path),
            "num_examples":len(predictions),
        },
        args.output_path.with_suffix(".metrics.json"),
    )

    if not args.no_wandb and wandb.run is not None:
        wandb.log(metrics)
        wandb.finish()

if __name__ == "__main__":
    main()