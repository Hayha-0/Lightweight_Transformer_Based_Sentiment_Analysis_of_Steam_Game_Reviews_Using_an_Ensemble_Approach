"""
train_qwen.py — Qwen3-0.6B (dense decoder, GQA + qk-layernorm)

Qwen3ForSequenceClassification ships natively in transformers>=4.51.0
(you have 5.14.x, so no custom code / trust_remote_code needed). Requires a
pad token to be set manually since Qwen3-0.6B's base tokenizer doesn't
define one by default.

Run:  python train_qwen.py
"""

import os
import sys
import time
import json
import torch
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, DataCollatorWithPadding, EarlyStoppingCallback,
)

sys.path.append(os.path.join(os.path.dirname(__file__)))
from common.data_utils import load_splits, build_hf_datasets, make_tokenize_fn, compute_class_weights
from common.metrics_utils import compute_metrics, WeightedTrainer, save_history_plot, full_test_evaluation
from common.model_utils import get_bnb_config, build_lora_model, print_trainable_parameters

# ---------------------------------------------------------------------------
MODEL_NAME = "qwen06b"
MODEL_ID = "Qwen/Qwen3-0.6B"
USE_4BIT = True
MAX_LENGTH = 512
NUM_LABELS = 2
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

OUTPUT_DIR = os.path.join("models", MODEL_NAME)
RESULTS_DIR = os.path.join("results", MODEL_NAME)
# ---------------------------------------------------------------------------


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        torch.cuda.reset_peak_memory_stats()

    print("Loading tokenizer/model:", MODEL_ID)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quant_config = get_bnb_config() if USE_4BIT else None
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_ID,
        num_labels=NUM_LABELS,
        quantization_config=quant_config,
        device_map="auto",
        torch_dtype=torch.bfloat16 if not USE_4BIT else None,
    )
    model.config.pad_token_id = tokenizer.pad_token_id

    model = build_lora_model(model, TARGET_MODULES, use_4bit=USE_4BIT)  # modules_to_save=("score",) by default
    print_trainable_parameters(model)
    # Sanity check: confirm the classification head is actually trainable, not silently frozen
    for name, p in model.named_parameters():
        if "score" in name and p.requires_grad:
            print(f"  head param trainable: {name}  shape={tuple(p.shape)}")

    print("Loading data splits...")
    train_df, val_df, test_df = load_splits()
    class_weights = compute_class_weights(train_df)
    print("Class weights [neg, pos]:", class_weights.tolist())

    train_ds, val_ds, test_ds = build_hf_datasets(train_df, val_df, test_df)
    tokenize_fn = make_tokenize_fn(tokenizer, MAX_LENGTH)
    train_ds = train_ds.map(tokenize_fn, batched=True, remove_columns=["text"])
    val_ds_tok = val_ds.map(tokenize_fn, batched=True, remove_columns=["text"])
    test_ds_tok = test_ds.map(tokenize_fn, batched=True, remove_columns=["text"])

    collator = DataCollatorWithPadding(tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=16,
        gradient_accumulation_steps=4,
        num_train_epochs=20,
        learning_rate=2e-4,
        weight_decay=0.01,
        warmup_ratio=0.06,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        bf16=True,
        gradient_checkpointing=True,
        logging_steps=25,
        report_to="none",
        dataloader_num_workers=2,
    )

    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds_tok,
        data_collator=collator,
        compute_metrics=compute_metrics,
        class_weights=class_weights,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=11)],
    )

    print("Starting training...")
    start = time.time()
    trainer.train()
    train_wall_time = time.time() - start
    print(f"Training complete in {train_wall_time/60:.1f} minutes")

    trainer.save_model(os.path.join(OUTPUT_DIR, "best_adapter"))
    tokenizer.save_pretrained(os.path.join(OUTPUT_DIR, "best_adapter"))

    save_history_plot(trainer.state.log_history, RESULTS_DIR, MODEL_NAME)

    print("Evaluating on held-out test set...")
    metrics = full_test_evaluation(
        trainer, test_ds_tok, RESULTS_DIR, MODEL_NAME, raw_texts=test_ds["text"]
    )
    metrics["training_wall_time_sec"] = train_wall_time
    with open(os.path.join(RESULTS_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print("Done. Results saved to:", RESULTS_DIR)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
