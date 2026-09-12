"""
evaluate.py
Reloads a saved LoRA adapter (from models/<name>/best_adapter) and re-runs
the full test-set evaluation, without retraining. Useful if you need to
regenerate metrics/figures/predictions.csv later, or evaluate on a
different held-out set.

Usage (PowerShell):
    python evaluate.py --model eurobert210m
    python evaluate.py --model gemma270m
    python evaluate.py --model qwen06b
"""

import os
import sys
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, DataCollatorWithPadding
from peft import PeftModel

sys.path.append(os.path.join(os.path.dirname(__file__)))
from common.data_utils import load_splits, build_hf_datasets, make_tokenize_fn
from common.metrics_utils import compute_metrics, full_test_evaluation
from common.model_utils import get_bnb_config

MODEL_REGISTRY = {
    "eurobert210m": dict(base_id="EuroBERT/EuroBERT-210m", trust_remote_code=True),
    "gemma270m": dict(base_id="google/gemma-3-270m", trust_remote_code=False),
    "qwen06b": dict(base_id="Qwen/Qwen3-0.6B", trust_remote_code=False),
}

MAX_LENGTH = 512
NUM_LABELS = 2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=list(MODEL_REGISTRY.keys()))
    parser.add_argument("--use_4bit", action="store_true", default=True)
    args = parser.parse_args()

    cfg = MODEL_REGISTRY[args.model]
    adapter_dir = os.path.join("models", args.model, "best_adapter")
    results_dir = os.path.join("results", args.model)
    os.makedirs(results_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(adapter_dir, trust_remote_code=cfg["trust_remote_code"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quant_config = get_bnb_config() if args.use_4bit else None
    base_model = AutoModelForSequenceClassification.from_pretrained(
        cfg["base_id"],
        num_labels=NUM_LABELS,
        quantization_config=quant_config,
        device_map="auto",
        trust_remote_code=cfg["trust_remote_code"],
    )
    base_model.config.pad_token_id = tokenizer.pad_token_id
    model = PeftModel.from_pretrained(base_model, adapter_dir)
    model.eval()

    _, _, test_df = load_splits()
    _, _, test_ds = build_hf_datasets(test_df, test_df, test_df)  # only test_ds is used below
    tokenize_fn = make_tokenize_fn(tokenizer, MAX_LENGTH)
    test_ds_tok = test_ds.map(tokenize_fn, batched=True, remove_columns=["text"])

    collator = DataCollatorWithPadding(tokenizer=tokenizer)
    dummy_args = TrainingArguments(
        output_dir=os.path.join("models", args.model, "_eval_tmp"),
        per_device_eval_batch_size=16,
        report_to="none",
    )
    trainer = Trainer(model=model, args=dummy_args, data_collator=collator, compute_metrics=compute_metrics)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    metrics = full_test_evaluation(trainer, test_ds_tok, results_dir, args.model, raw_texts=test_ds["text"])
    print(metrics)


if __name__ == "__main__":
    main()
