"""
kfold_best_model.py
Run this ONLY on the single best-performing model from generate_graphs.py.
Performs Stratified K-Fold cross-validation (default k=5) on train+validation
combined (test.csv stays held out and untouched, exactly like the rest of
the project), retraining a fresh LoRA adapter per fold, and reports mean +/-
std of every metric across folds. This is what you'll cite as the robustness
check before building the hybrid.

Edit BEST_MODEL below to the winning model's key from MODEL_REGISTRY, then:
    python kfold_best_model.py
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, DataCollatorWithPadding, EarlyStoppingCallback,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

sys.path.append(os.path.join(os.path.dirname(__file__)))
from common.data_utils import load_splits, normalize_labels, TEXT_COLUMN, LABEL_COLUMN, make_tokenize_fn
from common.metrics_utils import compute_metrics, WeightedTrainer, full_test_evaluation
from common.model_utils import get_bnb_config, print_trainable_parameters
from datasets import Dataset

# ---------------------------------------------------------------------------
BEST_MODEL = "qwen06b"     # <-- Qwen3-0.6B: second half of the Gemma+Qwen ensemble,
                           # k-fold run to validate the full ensemble (not just Gemma).
                           # Qwen's own convergence data (history.json) confirmed: real
                           # peak at epoch 6, never beaten through epoch 17 (where the
                           # original 20-epoch/patience-11 run actually stopped).
                           # patience=3 checked against Qwen's real epoch-1->2 dip
                           # (survives) and epoch-6->9 plateau (correctly stops ~epoch 9
                           # while still keeping the true epoch-6 best checkpoint via
                           # load_best_model_at_end). Same epoch=14 ceiling kept as a
                           # safety margin in case any fold's data mix converges later
                           # than this reference run did - see conversation notes.
N_FOLDS = 3   # reduced from the conventional 5-fold due to compute constraints;
              # still uses StratifiedKFold (not plain KFold) to preserve the
              # ~86%/14% positive/negative class ratio in every fold, given
              # the dataset's known class imbalance
MAX_LENGTH = 512
NUM_LABELS = 2
USE_4BIT = True

MODEL_REGISTRY = {
    "eurobert210m": dict(base_id="EuroBERT/EuroBERT-210m", trust_remote_code=True,
                          target_modules="all-linear", modules_to_save=["dense", "classifier"]),
    "gemma270m": dict(base_id="google/gemma-3-270m", trust_remote_code=False,
                       target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                       modules_to_save=["score"]),
    "qwen06b": dict(base_id="Qwen/Qwen3-0.6B", trust_remote_code=False,
                     target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                     modules_to_save=["score"]),
}
# ---------------------------------------------------------------------------


def build_model(cfg, tokenizer):
    if cfg.get("trust_remote_code"):
        # EuroBERT-specific transformers 5.x RoPE compatibility patch — see
        # train_eurobert.py's module docstring for the confirmed root cause.
        from transformers import modeling_rope_utils
        if "default" not in modeling_rope_utils.ROPE_INIT_FUNCTIONS:
            default_fn = getattr(modeling_rope_utils, "_compute_default_rope_parameters", None)
            if default_fn is None:
                def default_fn(config=None, device=None, seq_len=None, **rope_kwargs):
                    base = getattr(config, "rope_theta", 10000.0) if config is not None else rope_kwargs.get("base", 10000.0)
                    prf = getattr(config, "partial_rotary_factor", 1.0) if config is not None else 1.0
                    head_dim = getattr(config, "head_dim", None) or (config.hidden_size // config.num_attention_heads)
                    dim = int(head_dim * prf)
                    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / dim))
                    return inv_freq, 1.0
            modeling_rope_utils.ROPE_INIT_FUNCTIONS["default"] = default_fn

    quant_config = get_bnb_config() if USE_4BIT else None
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg["base_id"], num_labels=NUM_LABELS, quantization_config=quant_config,
        device_map="auto", trust_remote_code=cfg["trust_remote_code"],
    )
    if tokenizer.pad_token_id is not None:
        model.config.pad_token_id = tokenizer.pad_token_id
    if USE_4BIT:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    lora_config = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05,
        target_modules=cfg["target_modules"],
        modules_to_save=cfg.get("modules_to_save"),
        bias="none", task_type="SEQ_CLS",
    )
    model = get_peft_model(model, lora_config)
    return model


def main():
    cfg = MODEL_REGISTRY[BEST_MODEL]
    print(f"Running {N_FOLDS}-fold CV on: {BEST_MODEL} ({cfg['base_id']})")

    train_df, val_df, _ = load_splits()
    combined = pd.concat([train_df, val_df], ignore_index=True)
    combined = normalize_labels(combined)

    tokenizer = AutoTokenizer.from_pretrained(cfg["base_id"], trust_remote_code=cfg["trust_remote_code"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenize_fn = make_tokenize_fn(tokenizer, MAX_LENGTH)
    collator = DataCollatorWithPadding(tokenizer=tokenizer)

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    fold_metrics = []

    os.makedirs("results/kfold_" + BEST_MODEL, exist_ok=True)

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(combined[TEXT_COLUMN], combined[LABEL_COLUMN])):
        print(f"\n===== Fold {fold_idx + 1}/{N_FOLDS} =====")
        fold_train = combined.iloc[train_idx].reset_index(drop=True)
        fold_val = combined.iloc[val_idx].reset_index(drop=True)

        counts = fold_train[LABEL_COLUMN].value_counts().sort_index()
        weights = (len(fold_train) / (len(counts) * counts)).sort_index().values.astype(np.float32)
        class_weights = torch.tensor(weights)

        train_ds = Dataset.from_pandas(
            fold_train[[TEXT_COLUMN, LABEL_COLUMN]].rename(columns={TEXT_COLUMN: "text", LABEL_COLUMN: "labels"}),
            preserve_index=False)
        val_ds = Dataset.from_pandas(
            fold_val[[TEXT_COLUMN, LABEL_COLUMN]].rename(columns={TEXT_COLUMN: "text", LABEL_COLUMN: "labels"}),
            preserve_index=False)
        val_texts = val_ds["text"]
        train_ds = train_ds.map(tokenize_fn, batched=True, remove_columns=["text"])
        val_ds_tok = val_ds.map(tokenize_fn, batched=True, remove_columns=["text"])

        model = build_model(cfg, tokenizer)
        print_trainable_parameters(model)

        fold_out_dir = f"models/kfold_{BEST_MODEL}/fold_{fold_idx}"
        training_args = TrainingArguments(
            output_dir=fold_out_dir,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=16,
            gradient_accumulation_steps=4,
            num_train_epochs=14,  # ceiling based on original Gemma run's convergence
                                  # evidence: val F1 peaked at epoch 10, tied through
                                  # epoch 14, declined after. 14 gives folds room to
                                  # peak slightly later than the reference run without
                                  # paying for epochs 15-20, which only showed decline.
            learning_rate=2e-4,
            eval_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=1,
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            greater_is_better=True,
            bf16=True,
            gradient_checkpointing=True,
            logging_steps=50,
            report_to="none",
        )

        trainer = WeightedTrainer(
            model=model, args=training_args, train_dataset=train_ds, eval_dataset=val_ds_tok,
            data_collator=collator, compute_metrics=compute_metrics, class_weights=class_weights,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
            # patience=3, not 2: the original Gemma run had a genuine 2-epoch dip
            # (epochs 4-5) before recovering and climbing to its real peak at
            # epoch 10. Patience=2 would have falsely stopped that exact run at
            # epoch 5, missing the true best checkpoint entirely - verified
            # directly against that run's logged eval_f1 history.
        )
        trainer.train()

        fold_results_dir = f"results/kfold_{BEST_MODEL}/fold_{fold_idx}"
        m = full_test_evaluation(trainer, val_ds_tok, fold_results_dir, f"{BEST_MODEL}_fold{fold_idx}", raw_texts=val_texts)
        fold_metrics.append(m)

        del model, trainer
        torch.cuda.empty_cache()

    summary = {
        metric: {
            "mean": float(np.mean([m[metric] for m in fold_metrics])),
            "std": float(np.std([m[metric] for m in fold_metrics])),
        }
        for metric in ["accuracy", "precision", "recall", "f1", "roc_auc"]
    }
    with open(f"results/kfold_{BEST_MODEL}/kfold_summary.json", "w") as f:
        json.dump({"per_fold": fold_metrics, "summary": summary}, f, indent=2)

    print("\n===== K-FOLD SUMMARY =====")
    for k, v in summary.items():
        print(f"{k}: {v['mean']:.4f} ± {v['std']:.4f}")


if __name__ == "__main__":
    main()
