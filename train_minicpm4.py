"""
train_minicpm4.py — MiniCPM4-0.5B (custom decoder, InfLLM v2 sparse attention)

READ THIS FIRST:

1. transformers 5.x compatibility patch (confirmed via
   test_minicpm4_compatibility.py — see patch below): MiniCPM4's remote
   modeling code imports `is_torch_fx_available` from
   transformers.utils.import_utils, which has been removed/renamed in
   transformers 5.x. Same category of issue as the EuroBERT RoPE patch —
   a version-compatibility gap in the remote code, not a real architecture
   incompatibility. Must be applied BEFORE the model is imported, i.e.
   before AutoModelForSequenceClassification.from_pretrained(trust_remote_code=True).

2. trust_remote_code=True is required on BOTH AutoTokenizer and
   AutoModelForSequenceClassification calls (same as EuroBERT).

3. Classification head is named "score" (a single nn.Linear, in=1024,
   out=2) — confirmed via test_minicpm4_compatibility.py's Step 3. This
   matches the Qwen/Gemma convention rather than EuroBERT's split
   "dense"/"classifier" pair.

4. IMPORTANT — 4-bit quantization risk carried over from the EuroBERT bug:
   the reason Qwen/Gemma's "score" head is auto-excluded from 4-bit
   conversion is that they're transformers-native architectures, not
   because the head happens to be named "score". MiniCPM4 loads through
   trust_remote_code, same as EuroBERT — so that auto-exclusion is NOT
   guaranteed here even though the head has the "safe" name. To avoid
   silently reproducing the EuroBERT AssertionError
   (module.weight.shape[1] == 1 / "FP4 quantization state not
   initialized"), this script defensively passes
   skip_modules=["score"] to get_bnb_config(), exactly like
   train_eurobert.py does for "dense"/"classifier". If this later proves
   unnecessary (auto-exclusion does apply here), it's a no-op — passing
   skip_modules for a module that would've been excluded anyway is
   harmless.

5. target_modules="all-linear" is used (not a guessed Llama-style list)
   because MiniCPM4 is custom/trust_remote_code — same reasoning as
   EuroBERT: linear-layer naming isn't guaranteed to match
   q_proj/k_proj/v_proj/o_proj.

6. use_cache is forced to False, matching the finding in
   test_minicpm4_compatibility.py (Step 4 comment) that MiniCPM4's remote
   code still emits the old tuple-based past_key_values format, which
   your installed transformers version no longer accepts. Not relevant
   for classification anyway (single forward pass, no generation).

Run:  python train_minicpm4.py
"""

import os
import sys
import time
import json
import torch

# PATCH: must run before MiniCPM4's remote modeling code is imported
# (i.e. before AutoModelForSequenceClassification.from_pretrained below).
# See module docstring, point 1.
import transformers.utils.import_utils as _iu
if not hasattr(_iu, "is_torch_fx_available"):
    _iu.is_torch_fx_available = lambda: False
    print("Patched transformers.utils.import_utils.is_torch_fx_available for MiniCPM4 compatibility")

from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, DataCollatorWithPadding, EarlyStoppingCallback,
)

sys.path.append(os.path.join(os.path.dirname(__file__)))
from common.data_utils import load_splits, build_hf_datasets, make_tokenize_fn, compute_class_weights
from common.metrics_utils import compute_metrics, WeightedTrainer, save_history_plot, full_test_evaluation
from common.model_utils import get_bnb_config, print_trainable_parameters, list_linear_module_names
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# ---------------------------------------------------------------------------
MODEL_NAME = "minicpm4_05b"
MODEL_ID = "openbmb/MiniCPM4-0.5B"
USE_4BIT = True
MAX_LENGTH = 512
NUM_LABELS = 2
RUN_MODULE_DISCOVERY_ONLY = False   # set True once to just print Linear layer names and exit

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
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # See module docstring, point 4: defensive skip_modules even though the
    # head name ("score") matches the "safe" convention, because auto-
    # exclusion from 4-bit conversion is tied to native-architecture status,
    # not to the head's name, and MiniCPM4 is trust_remote_code like EuroBERT.
    quant_config = get_bnb_config(skip_modules=["score"]) if USE_4BIT else None
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_ID,
        num_labels=NUM_LABELS,
        quantization_config=quant_config,
        device_map="auto",
        torch_dtype=torch.bfloat16 if not USE_4BIT else None,
        trust_remote_code=True,
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False  # see module docstring, point 6

    if RUN_MODULE_DISCOVERY_ONLY:
        list_linear_module_names(model)
        return

    if USE_4BIT:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules="all-linear",   # custom/remote-code arch — don't guess module names
        # "score" is newly initialized (missing from the pretrained checkpoint
        # per the LOAD REPORT in test_minicpm4_compatibility.py), so it must be
        # fully trained rather than LoRA-decomposed, and excluded from
        # "all-linear"'s wrapping so PEFT doesn't try to bnb-dispatch a
        # non-quantized layer. Same reasoning as EuroBERT's modules_to_save.
        modules_to_save=["score"],
        bias="none",
        task_type="SEQ_CLS",
    )
    model = get_peft_model(model, lora_config)
    print_trainable_parameters(model)
    # Sanity check: confirm the head is actually trainable, not silently frozen
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
