"""
train_eurobert.py — EuroBERT-210m (bidirectional multilingual encoder)

READ THIS FIRST — transformers 5.x compatibility patch:

EuroBERT's `trust_remote_code` modeling file (modeling_eurobert.py) crashes
on transformers 5.x with:
    KeyError: 'default'
    ...in EuroBertRotaryEmbedding.__init__ -> ROPE_INIT_FUNCTIONS[self.rope_type]

ROOT CAUSE (confirmed, not guessed): transformers 5.0 restructured RoPE
handling (config.rope_scaling/rope_theta -> unified config.rope_parameters)
and, as part of that refactor, removed the 'default' key from the internal
ROPE_INIT_FUNCTIONS dispatch dict in transformers.modeling_rope_utils.
EuroBERT's remote code was written against the pre-5.0 API and still expects
that key to exist. This is an independently-reported failure mode — the
Qwen3-TTS project hit and patched the identical KeyError for the same reason.

FIX: patch_eurobert_rope_compatibility() below restores the 'default' key
before EuroBERT's code is imported (which happens inside
AutoModelForSequenceClassification.from_pretrained(..., trust_remote_code=True)).
It does not change model behavior — it just re-registers the standard,
unscaled RoPE frequency formula under the key EuroBERT's code looks up.

OTHER NOTES (from earlier debugging):
- trust_remote_code=True is required on BOTH AutoTokenizer and
  AutoModelForSequenceClassification calls.
- Linear-layer naming inside EuroBERT isn't guaranteed to match Llama-style
  q_proj/k_proj/v_proj/o_proj, so LoRA uses target_modules="all-linear"
  (peft>=0.10) instead of a guessed module list.
- No pad-token workaround needed — EuroBERT's tokenizer ships a real [PAD].
- Consider pinning a specific commit hash instead of "main" for
  reproducibility, e.g. MODEL_ID = "EuroBERT/EuroBERT-210m@<commit_sha>",
  since trust_remote_code executes whatever code currently lives in the repo.

Run:  python train_eurobert.py
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
from common.model_utils import get_bnb_config, print_trainable_parameters, list_linear_module_names
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# ---------------------------------------------------------------------------
MODEL_NAME = "eurobert210m"
MODEL_ID = "EuroBERT/EuroBERT-210m"     # swap to "EuroBERT/EuroBERT-610m" for the larger variant
USE_4BIT = True
MAX_LENGTH = 512
NUM_LABELS = 2
RUN_MODULE_DISCOVERY_ONLY = False   # set True once to just print Linear layer names and exit

OUTPUT_DIR = os.path.join("models", MODEL_NAME)
RESULTS_DIR = os.path.join("results", MODEL_NAME)
# ---------------------------------------------------------------------------


def patch_eurobert_rope_compatibility():
    """Restore the 'default' ROPE_INIT_FUNCTIONS key removed in transformers 5.0.
    See module docstring above for the confirmed root cause. Safe to call
    even on transformers versions where 'default' already exists (no-op)."""
    from transformers import modeling_rope_utils

    if "default" in modeling_rope_utils.ROPE_INIT_FUNCTIONS:
        return

    default_fn = getattr(modeling_rope_utils, "_compute_default_rope_parameters", None)

    if default_fn is None:
        # Fallback: the function itself was removed too, not just unregistered.
        # This re-implements the standard, well-documented unscaled RoPE formula
        # (same one every Llama-family model uses) rather than guessing at
        # transformers' internal API.
        def default_fn(config=None, device=None, seq_len=None, **rope_kwargs):
            base = getattr(config, "rope_theta", 10000.0) if config is not None else rope_kwargs.get("base", 10000.0)
            partial_rotary_factor = getattr(config, "partial_rotary_factor", 1.0) if config is not None else 1.0
            head_dim = getattr(config, "head_dim", None) or (config.hidden_size // config.num_attention_heads)
            dim = int(head_dim * partial_rotary_factor)
            inv_freq = 1.0 / (
                base ** (torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / dim)
            )
            attention_factor = 1.0
            return inv_freq, attention_factor

    modeling_rope_utils.ROPE_INIT_FUNCTIONS["default"] = default_fn
    print("Patched transformers.modeling_rope_utils.ROPE_INIT_FUNCTIONS['default'] "
          "for EuroBERT compatibility with transformers 5.x")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    patch_eurobert_rope_compatibility()

    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        torch.cuda.reset_peak_memory_stats()

    print("Loading tokenizer/model:", MODEL_ID)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

    # FIX (confirmed root cause): without skip_modules, from_pretrained()
    # 4-bit-converts EVERY nn.Linear including the newly-initialized
    # "dense"/"classifier" head. modules_to_save then deep-copies that
    # already-quantized Linear4bit layer to make it trainable, the copy's
    # Params4bit state is never properly initialized, and the first
    # forward pass crashes in bitsandbytes with
    # AssertionError: module.weight.shape[1] == 1 (preceded by the
    # "FP4 quantization state not initialized" warning). Passing
    # skip_modules here keeps "dense"/"classifier" as plain nn.Linear in
    # bf16/fp32, exactly matching what happens automatically for Qwen's
    # "score" head on transformers-native architectures. See
    # common/model_utils.py:get_bnb_config() docstring for full detail.
    quant_config = get_bnb_config(skip_modules=["dense", "classifier"]) if USE_4BIT else None
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_ID,
        num_labels=NUM_LABELS,
        quantization_config=quant_config,
        device_map="auto",
        torch_dtype=torch.bfloat16 if not USE_4BIT else None,
        trust_remote_code=True,
    )

    if RUN_MODULE_DISCOVERY_ONLY:
        list_linear_module_names(model)
        return

    if USE_4BIT:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules="all-linear",
        # The classification head ("dense" + "classifier") is newly initialized
        # (see the LOAD REPORT above — those weights are MISSING from the
        # pretrained masked-LM checkpoint), so it's a plain, unquantized
        # nn.Linear even though the rest of the model is 4-bit. Two things
        # follow: (1) it must be fully trained, not LoRA-decomposed, since it
        # starts random; (2) it must be EXCLUDED from "all-linear"'s LoRA
        # wrapping, or PEFT tries to bnb-4bit-dispatch a non-quantized layer
        # and crashes with AttributeError: 'Parameter' object has no
        # attribute 'compress_statistics'. modules_to_save handles both.
        modules_to_save=["dense", "classifier"],
        bias="none",
        task_type="SEQ_CLS",
    )
    model = get_peft_model(model, lora_config)
    print_trainable_parameters(model)
    # Sanity check: confirm the head is actually trainable now, not silently frozen
    for name, p in model.named_parameters():
        if ("classifier" in name or name.endswith("dense.weight") or name.endswith("dense.bias")) and p.requires_grad:
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
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        gradient_accumulation_steps=2,
        num_train_epochs=20,
        learning_rate=3e-4,
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
