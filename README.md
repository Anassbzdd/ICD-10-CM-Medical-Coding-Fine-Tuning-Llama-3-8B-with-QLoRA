# ICD-10 Medical Coding with Llama 3 8B QLoRA

Fine-tuned `meta-llama/Meta-Llama-3-8B` with QLoRA to predict ICD-10-CM diagnosis codes from synthetic EHR clinical notes. The goal of this project is to show an end-to-end ML engineering workflow: dataset cleaning, prompt construction, memory-efficient fine-tuning, checkpoint recovery, evaluation, inference, and a FastAPI serving endpoint.

> This project is for ML engineering education and portfolio demonstration only. It is not a medical device and should not be used for real clinical coding decisions.

## Results

The final model was trained on 50,000 cleaned synthetic EHR examples and evaluated with generation-based ICD-10 code extraction.

| Split | Examples | Precision | Recall | F1 | Exact Match | Micro F1 |
|---|---:|---:|---:|---:|---:|---:|
| Validation | 2,000 | 0.8705 | 0.8705 | 0.8705 | 0.8705 | 0.8714 |
| Test | 500 | 0.8720 | 0.8720 | 0.8720 | 0.8720 | 0.8755 |

Training summary:

| Item | Value |
|---|---|
| Base model | `meta-llama/Meta-Llama-3-8B` |
| Fine-tuning method | QLoRA |
| Train samples | 50,000 |
| Eval samples during training | 2,000 |
| Max sequence length | 768 |
| Final train loss | 0.064 |
| Trainable params | 13.6M LoRA params |

## What This Project Demonstrates

- Fine-tuning an open-weight LLM instead of wrapping an API.
- 4-bit QLoRA training with `bitsandbytes`, `peft`, `transformers`, and `trl`.
- LoRA adapters on Llama attention projections: `q_proj`, `k_proj`, `v_proj`, `o_proj`.
- Completion-only supervised fine-tuning so loss is applied only to ICD-code outputs, not prompt tokens.
- Dataset cleaning, leakage filtering, deterministic train/validation/test splitting, and report generation.
- Resume-safe Kaggle training with Hugging Face Hub checkpoint uploads.
- Offline generation evaluation with precision, recall, F1, exact match, and micro metrics.
- CLI inference and FastAPI serving.

## Dataset

Hugging Face dataset:

```text
generative-technologies/synth-ehr-icd10-llama3-format
```

After preprocessing, the final split sizes were:

```text
train:      206,226
validation: 11,457
test:       11,458
```

Each raw row is cleaned into a structured training example:

```python
{
    "example_id": "...",
    "ehr_text": "Synthetic clinical note...",
    "instruction": "Task instruction...",
    "raw_icd_code": "...",
    "raw_output": "...",
    "target_text": "E11.9, I10",
    "prompt_text": "<|begin_of_text|>...<|start_header_id|>assistant<|end_header_id|>\n\n",
    "text": "<prompt_text>E11.9, I10<|eot_id|>",
    "num_codes": 2,
    "had_label_mismatch": False,
}
```

The training script then exposes modern TRL prompt/completion fields:

```python
prompt = example["prompt_text"]
completion = example["target_text"] + "<|eot_id|>"
```

With `completion_only_loss=True`, prompt tokens are masked with `-100`, so the model learns to generate only the ICD-10 answer.

## Training Approach

The project uses QLoRA to fit Llama 3 8B on limited Kaggle GPU memory.

Key choices:

```text
4-bit quantization: NF4
Double quantization: enabled
Compute dtype: float16
LoRA rank: 16
LoRA alpha: 32
LoRA dropout: 0.05
Target modules: q_proj, k_proj, v_proj, o_proj
Optimizer: paged_adamw_8bit
Gradient checkpointing: enabled
Trainer precision flags: fp16=False, bf16=False
```

Why this matters:

- The base model is loaded in 4-bit, reducing memory usage.
- Only LoRA adapter weights are trainable.
- `paged_adamw_8bit` reduces optimizer memory and helps avoid T4 out-of-memory errors.
- Completion-only loss focuses learning on ICD-code generation instead of copying the clinical note.

## Project Structure

```text
.
├── src/
│   ├── api.py              # FastAPI application
│   ├── config.py           # Dataclass configs and defaults
│   ├── evaluate.py         # Generation-based evaluation
│   ├── inference.py        # CLI inference
│   ├── metrics.py          # ICD extraction and metrics
│   ├── modeling.py         # Tokenizer, 4-bit model loading, PEFT setup
│   ├── preprocessing.py    # Dataset cleaning, leakage filtering, splitting
│   ├── prompting.py        # Manual Llama 3 prompt formatting
│   ├── train.py            # QLoRA training entrypoint
│   └── utils.py            # Logging, serialization, checkpoint helpers
├── data/
│   ├── processed/          # Saved Hugging Face DatasetDict
│   └── reports/            # Dataset reports
├── results/                # Prediction JSONL files and metrics
├── checkpoints/            # Local LoRA checkpoints/adapters
├── requirements.txt
└── README.md
```

## Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

Login to Hugging Face if using gated Llama weights or Hub checkpoint uploads:

```bash
huggingface-cli login
```

Optional W&B login:

```bash
wandb login
```

## Preprocess Data

```bash
python -m src.preprocessing \
  --processed_dir data/processed \
  --report_path data/reports/dataset_report.json \
  --log_level INFO
```

This step:

- Loads the Hugging Face dataset.
- Normalizes ICD-code text.
- Filters empty examples.
- Filters examples where ICD codes leak into the clinical note or prompt.
- Deduplicates examples.
- Builds Llama 3 formatted prompt/completion training text.
- Saves a reproducible `DatasetDict`.

## Train

Recommended Kaggle command:

```bash
PYTHONPATH=/kaggle/working/project python -m src.train \
  --output_dir /kaggle/working/project/checkpoints/final_run_50k \
  --run_name llama3_8b_qlora_icd10_50k \
  --max_train_samples 50000 \
  --max_eval_samples 2000 \
  --max_seq_length 768 \
  --num_train_epochs 1 \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --gradient_accumulation_steps 16 \
  --learning_rate 2e-4 \
  --warmup_ratio 0.03 \
  --weight_decay 0.01 \
  --logging_steps 10 \
  --eval_steps 250 \
  --save_steps 50 \
  --save_total_limit 8 \
  --push_to_hub \
  --hub_model_id "$HF_REPO_ID" \
  --hub_strategy checkpoint \
  --hub_always_push
```

The final adapter is saved under:

```text
checkpoints/final_run_50k/last-checkpoint
```

This is a LoRA adapter, not a full copy of Llama 3.

## Resume Training

Kaggle resets can delete `/kaggle/working`, so the project pushes checkpoints to Hugging Face Hub during training.

After a reset:

```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="YOUR_USERNAME/icd10-llama3-8b-qlora-checkpoints",
    repo_type="model",
    local_dir="/kaggle/working/project/checkpoints/final_run_50k",
    allow_patterns=["last-checkpoint/*"],
)
```

Then resume:

```bash
PYTHONPATH=/kaggle/working/project python -m src.train \
  --output_dir /kaggle/working/project/checkpoints/final_run_50k \
  --resume_from_checkpoint /kaggle/working/project/checkpoints/final_run_50k/last-checkpoint \
  --push_to_hub \
  --hub_model_id "$HF_REPO_ID" \
  --hub_strategy checkpoint \
  --hub_always_push
```

## Evaluate

Validation:

```bash
PYTHONPATH=/kaggle/working/project python -m src.evaluate \
  --adapter_path /kaggle/working/project/checkpoints/final_run_50k/last-checkpoint \
  --processed_dir /kaggle/working/project/data/processed \
  --split validation \
  --max_samples 2000 \
  --batch_size 4 \
  --max_new_tokens 32 \
  --num_beams 1 \
  --output_path /kaggle/working/project/results/final_run_50k_validation_predictions.jsonl \
  --no_wandb
```

Test:

```bash
TRANSFORMERS_VERBOSITY=error PYTHONPATH=/kaggle/working/project python -m src.evaluate \
  --adapter_path /kaggle/working/project/checkpoints/final_run_50k/last-checkpoint \
  --processed_dir /kaggle/working/project/data/processed \
  --split test \
  --max_samples 500 \
  --batch_size 8 \
  --max_new_tokens 24 \
  --num_beams 1 \
  --output_path /kaggle/working/project/results/final_run_50k_test_500_predictions.jsonl \
  --no_wandb
```

Metrics are saved next to the prediction file:

```text
results/final_run_50k_test_500_predictions.metrics.json
```

## Inference

Run one clinical note from the CLI:

```bash
python -m src.inference \
  --adapter_path checkpoints/final_run_50k/last-checkpoint \
  --note "Patient is a 58-year-old male with type 2 diabetes mellitus without complications and essential hypertension." \
  --max_new_tokens 32 \
  --num_beams 1
```

## FastAPI Serving

Start the API:

```bash
python -m src.api \
  --adapter_path checkpoints/final_run_50k/last-checkpoint \
  --host 0.0.0.0 \
  --port 8000
```

Example request:

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "clinical_note": "Patient has type 2 diabetes mellitus without complications and essential hypertension.",
    "max_new_tokens": 32
  }'
```

## Important Implementation Details

Manual Llama 3 prompt format is used instead of relying on `tokenizer.chat_template`, because the base Llama 3 tokenizer may not expose a chat template in some environments.

```text
<|begin_of_text|>
<|start_header_id|>system<|end_header_id|>

...
<|eot_id|>
<|start_header_id|>user<|end_header_id|>

...
<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>

ICD_CODES<|eot_id|>
```

The tokenizer pad token is set to EOS for Llama compatibility:

```python
tokenizer.pad_token = tokenizer.eos_token
```

The project uses modern TRL:

```python
SFTConfig(completion_only_loss=True)
```

Older examples often use `DataCollatorForCompletionOnlyLM`; this project avoids that deprecated/import-fragile path for newer TRL versions.

## Limitations

- The dataset is synthetic, so performance does not prove readiness for real clinical deployment.
- Evaluation uses ICD-code extraction from generated text; raw generations may contain extra formatting that should be cleaned more aggressively for production.
- The final artifact is a LoRA adapter and requires the original gated Llama 3 base model to run.
- Test evaluation was run on 500 examples due to Kaggle runtime constraints; validation evaluation was run on 2,000 examples.
- The project should not be used for real healthcare decisions without clinical validation, privacy review, and regulatory work.

## Next Improvements

- Add stricter generation post-processing to remove malformed text around ICD codes.
- Push the processed dataset to a Hugging Face dataset repo so Kaggle resets do not require rebuilding it.
- Add an automated smoke-test workflow for preprocessing, inference, and API startup.
- Evaluate on more realistic multi-code notes and out-of-distribution clinical notes.
- Add Docker packaging for reproducible API deployment.

## Tech Stack

```text
Python
PyTorch
Hugging Face Transformers
TRL
PEFT
bitsandbytes
datasets
Weights & Biases
FastAPI
Kaggle T4 x2
Hugging Face Hub
```
