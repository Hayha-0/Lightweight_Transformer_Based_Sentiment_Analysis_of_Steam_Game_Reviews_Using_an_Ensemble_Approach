"""
benchmark_efficiency_repeated.py -- runs the full 5-way efficiency
benchmark (EuroBERT-210M, MiniCPM4-0.5B, Qwen3-0.6B, Gemma 3 270M,
Ensemble) N_TRIALS times back-to-back in one execution, reporting
mean +/- std per model for latency and peak VRAM -- replacing the
single-run point estimate in efficiency_benchmark.csv with a
properly quantified, repeated-measurement result.

WHY THIS EXISTS:
The original benchmark_efficiency.py runs each model exactly once.
That single-run point estimate is vulnerable to normal run-to-run
system noise (GPU clock/thermal state, background OS/driver
scheduling jitter, first-touch memory allocation variance, etc.),
and a single number with no variance reported is an easy target for
a reviewer to question ("how do you know that's representative?").
This script answers that directly: run the same measurement
multiple times, report mean +/- std, exactly like the paper's
existing Table~\ref{tab:cv} (3-fold CV mean +/- std) already does
for accuracy -- extending the same reporting standard to efficiency.

WHAT'S DIFFERENT FROM THE ORIGINAL SCRIPT:
- N_TRIALS full passes (default 3) over the SAME held-out test set
  (n=1,710) per model, not 3 different data samples -- this isolates
  measurement/system noise from data variance, which is the right
  isolation for a "how stable is this timing measurement" question
  (data variance is not in question here; the test set is fixed).
- Each trial reloads the model fresh (not just re-timed on an
  already-loaded model) -- this is deliberately the more
  conservative/realistic choice: a fresh model load captures
  cold-start allocation behavior each time rather than only
  measuring steady-state repeated inference on a model already
  warmed into a stable memory/kernel state. If you'd rather isolate
  ONLY inference-loop variance (load once, time N_TRIALS passes over
  the same loaded model), see the LOAD_ONCE_PER_MODEL flag below --
  set it to True to switch modes. Default is False (reload each
  trial) since that's the more defensible choice for a paper claim.
- Same GPU-synchronized timing, same warm-up-then-discard protocol,
  same peak-VRAM reset-per-measurement logic as the original script
  -- nothing about the measurement methodology itself changed,
  only the number of repetitions and the aggregation at the end.

OUTPUT:
- results/efficiency_benchmark_repeated_raw.csv
    One row per (model, trial) -- every individual trial's numbers,
    kept for full transparency/reproducibility.
- results/efficiency_benchmark_repeated_summary.csv
    One row per model -- mean and std of mean_latency_ms,
    median_latency_ms, and peak_vram_mb across the N_TRIALS trials.
    THIS is the file whose numbers go into the paper's table.

Run:  python benchmark_efficiency_repeated.py
Expect roughly N_TRIALS x (the original script's total runtime).
The original completed all 5 configs on the full 1,710-review test
set well within a few minutes total, so 3 trials should still
comfortably finish in well under 15-20 minutes.
"""
import os
import time
import json
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification, BitsAndBytesConfig
from peft import PeftModel

# ---------------------------------------------------------------------------
# COMPATIBILITY SHIM: same as the original script.
# ---------------------------------------------------------------------------
import transformers.utils.import_utils as _iu
if not hasattr(_iu, "is_torch_fx_available"):
    _iu.is_torch_fx_available = lambda: True

PROJECT_ROOT = Path(__file__).resolve().parent  # adjust if script location differs

TEST_CSV = str(PROJECT_ROOT / "data" / "test.csv")
MAX_LENGTH = 512
N_WARMUP = 5  # discarded from timing, just to prime CUDA kernels

N_TRIALS = 3               # <-- number of repeated full passes per model
LOAD_ONCE_PER_MODEL = False # <-- see note above; False = reload fresh each trial


def patch_eurobert_rope_compatibility():
    """Same patch as train_eurobert.py -- restores the 'default' ROPE_INIT_FUNCTIONS
    key removed in transformers 5.0, which EuroBERT's remote code still expects."""
    from transformers import modeling_rope_utils
    if "default" in modeling_rope_utils.ROPE_INIT_FUNCTIONS:
        return
    default_fn = getattr(modeling_rope_utils, "_compute_default_rope_parameters", None)
    if default_fn is None:
        def default_fn(config=None, device=None, seq_len=None, **rope_kwargs):
            base = getattr(config, "rope_theta", 10000.0) if config is not None else rope_kwargs.get("base", 10000.0)
            partial_rotary_factor = getattr(config, "partial_rotary_factor", 1.0) if config is not None else 1.0
            head_dim = getattr(config, "head_dim", None) or (config.hidden_size // config.num_attention_heads)
            dim = int(head_dim * partial_rotary_factor)
            inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / dim))
            return inv_freq, 1.0
    modeling_rope_utils.ROPE_INIT_FUNCTIONS["default"] = default_fn


MODELS = [
    {
        "name": "EuroBERT-210M",
        "base_id": "EuroBERT/EuroBERT-210m",
        "adapter_dir": str(PROJECT_ROOT / "models" / "eurobert210m" / "best_adapter"),
        "trust_remote_code": True,
        "needs_rope_patch": True,
        "bnb_skip_modules": ["dense", "classifier"],
        "use_cache": None,
    },
    {
        "name": "MiniCPM4-0.5B",
        "base_id": "openbmb/MiniCPM4-0.5B",
        "adapter_dir": str(PROJECT_ROOT / "models" / "minicpm4_05b" / "best_adapter"),
        "trust_remote_code": True,
        "needs_rope_patch": False,
        "bnb_skip_modules": None,
        "use_cache": False,
    },
    {
        "name": "Qwen3-0.6B",
        "base_id": "Qwen/Qwen3-0.6B",
        "adapter_dir": str(PROJECT_ROOT / "models" / "qwen06b" / "best_adapter"),
        "trust_remote_code": False,
        "needs_rope_patch": False,
        "bnb_skip_modules": None,
        "use_cache": None,
    },
    {
        "name": "Gemma 3 270M",
        "base_id": "google/gemma-3-270m",
        "adapter_dir": str(PROJECT_ROOT / "models" / "gemma270m" / "best_adapter"),
        "trust_remote_code": False,
        "needs_rope_patch": False,
        "bnb_skip_modules": None,
        "use_cache": None,
    },
]


def make_bnb_config(skip_modules):
    kwargs = dict(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    if skip_modules:
        kwargs["llm_int8_skip_modules"] = skip_modules
    return BitsAndBytesConfig(**kwargs)


def load_single_model(cfg):
    if cfg["needs_rope_patch"]:
        patch_eurobert_rope_compatibility()
    tokenizer = AutoTokenizer.from_pretrained(
        cfg["adapter_dir"], trust_remote_code=cfg["trust_remote_code"]
    )
    bnb_config = make_bnb_config(cfg.get("bnb_skip_modules"))
    base_model = AutoModelForSequenceClassification.from_pretrained(
        cfg["base_id"],
        num_labels=2,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=cfg["trust_remote_code"],
    )
    model = PeftModel.from_pretrained(base_model, cfg["adapter_dir"])
    model.eval()
    forward_kwargs = {}
    if cfg.get("use_cache") is False:
        model.config.use_cache = False
        forward_kwargs["use_cache"] = False
    return model, base_model, tokenizer, forward_kwargs


def time_single_model_pass(model, tokenizer, forward_kwargs, texts, device):
    torch.cuda.reset_peak_memory_stats(device)
    with torch.no_grad():
        for text in texts[:N_WARMUP]:
            inputs = tokenizer(text, truncation=True, max_length=MAX_LENGTH, return_tensors="pt").to(device)
            torch.cuda.synchronize()
            _ = model(**inputs, **forward_kwargs)
            torch.cuda.synchronize()

    latencies_ms = []
    with torch.no_grad():
        for text in texts:
            inputs = tokenizer(text, truncation=True, max_length=MAX_LENGTH, return_tensors="pt").to(device)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = model(**inputs, **forward_kwargs)
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            latencies_ms.append((t1 - t0) * 1000.0)

    peak_vram_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    return {
        "mean_latency_ms": sum(latencies_ms) / len(latencies_ms),
        "median_latency_ms": sorted(latencies_ms)[len(latencies_ms) // 2],
        "min_latency_ms": min(latencies_ms),
        "max_latency_ms": max(latencies_ms),
        "peak_vram_mb": peak_vram_mb,
    }


def benchmark_model_repeated(cfg, texts, n_trials):
    trial_results = []

    if LOAD_ONCE_PER_MODEL:
        print(f"\nLoading {cfg['name']} (once, reused across {n_trials} trials) ...")
        model, base_model, tokenizer, forward_kwargs = load_single_model(cfg)
        device = next(model.parameters()).device
        for trial in range(n_trials):
            r = time_single_model_pass(model, tokenizer, forward_kwargs, texts, device)
            r["model"] = cfg["name"]; r["trial"] = trial + 1; r["n_reviews"] = len(texts)
            trial_results.append(r)
            print(f"  [{cfg['name']}] trial {trial+1}/{n_trials}: "
                  f"mean={r['mean_latency_ms']:.2f}ms  peak_vram={r['peak_vram_mb']:.1f}MB")
        del model, base_model, tokenizer
        torch.cuda.empty_cache()
    else:
        for trial in range(n_trials):
            print(f"\nLoading {cfg['name']} (trial {trial+1}/{n_trials}, fresh load) ...")
            model, base_model, tokenizer, forward_kwargs = load_single_model(cfg)
            device = next(model.parameters()).device
            r = time_single_model_pass(model, tokenizer, forward_kwargs, texts, device)
            r["model"] = cfg["name"]; r["trial"] = trial + 1; r["n_reviews"] = len(texts)
            trial_results.append(r)
            print(f"  [{cfg['name']}] trial {trial+1}/{n_trials}: "
                  f"mean={r['mean_latency_ms']:.2f}ms  peak_vram={r['peak_vram_mb']:.1f}MB")
            del model, base_model, tokenizer
            torch.cuda.empty_cache()

    return trial_results


def benchmark_ensemble_once(gemma_cfg, qwen_cfg, texts):
    gemma_tok = AutoTokenizer.from_pretrained(gemma_cfg["adapter_dir"], trust_remote_code=gemma_cfg["trust_remote_code"])
    gemma_base = AutoModelForSequenceClassification.from_pretrained(
        gemma_cfg["base_id"], num_labels=2,
        quantization_config=make_bnb_config(gemma_cfg.get("bnb_skip_modules")),
        device_map="auto", trust_remote_code=gemma_cfg["trust_remote_code"],
    )
    gemma_model = PeftModel.from_pretrained(gemma_base, gemma_cfg["adapter_dir"]).eval()

    qwen_tok = AutoTokenizer.from_pretrained(qwen_cfg["adapter_dir"], trust_remote_code=qwen_cfg["trust_remote_code"])
    qwen_base = AutoModelForSequenceClassification.from_pretrained(
        qwen_cfg["base_id"], num_labels=2,
        quantization_config=make_bnb_config(qwen_cfg.get("bnb_skip_modules")),
        device_map="auto", trust_remote_code=qwen_cfg["trust_remote_code"],
    )
    qwen_model = PeftModel.from_pretrained(qwen_base, qwen_cfg["adapter_dir"]).eval()

    device = next(gemma_model.parameters()).device
    torch.cuda.reset_peak_memory_stats(device)
    with torch.no_grad():
        for text in texts[:N_WARMUP]:
            gi = gemma_tok(text, truncation=True, max_length=MAX_LENGTH, return_tensors="pt").to(device)
            torch.cuda.synchronize(); _ = gemma_model(**gi); torch.cuda.synchronize()
            qi = qwen_tok(text, truncation=True, max_length=MAX_LENGTH, return_tensors="pt").to(device)
            torch.cuda.synchronize(); _ = qwen_model(**qi); torch.cuda.synchronize()

    latencies_ms = []
    with torch.no_grad():
        for text in texts:
            gi = gemma_tok(text, truncation=True, max_length=MAX_LENGTH, return_tensors="pt").to(device)
            qi = qwen_tok(text, truncation=True, max_length=MAX_LENGTH, return_tensors="pt").to(device)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = gemma_model(**gi)
            _ = qwen_model(**qi)
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            latencies_ms.append((t1 - t0) * 1000.0)

    peak_vram_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)

    del gemma_model, gemma_base, gemma_tok, qwen_model, qwen_base, qwen_tok
    torch.cuda.empty_cache()

    return {
        "mean_latency_ms": sum(latencies_ms) / len(latencies_ms),
        "median_latency_ms": sorted(latencies_ms)[len(latencies_ms) // 2],
        "min_latency_ms": min(latencies_ms),
        "max_latency_ms": max(latencies_ms),
        "peak_vram_mb": peak_vram_mb,
    }


def benchmark_ensemble_repeated(gemma_cfg, qwen_cfg, texts, n_trials):
    trial_results = []
    for trial in range(n_trials):
        print(f"\nLoading ensemble constituents (trial {trial+1}/{n_trials}) ...")
        r = benchmark_ensemble_once(gemma_cfg, qwen_cfg, texts)
        r["model"] = "Ensemble (0.5G+0.5Q)"; r["trial"] = trial + 1; r["n_reviews"] = len(texts)
        trial_results.append(r)
        print(f"  [Ensemble] trial {trial+1}/{n_trials}: "
              f"mean={r['mean_latency_ms']:.2f}ms  peak_vram={r['peak_vram_mb']:.1f}MB")
    return trial_results


def summarize(raw_df):
    rows = []
    for model, grp in raw_df.groupby("model", sort=False):
        rows.append({
            "model": model,
            "n_trials": len(grp),
            "mean_latency_ms_mean": grp["mean_latency_ms"].mean(),
            "mean_latency_ms_std": grp["mean_latency_ms"].std(ddof=1) if len(grp) > 1 else 0.0,
            "median_latency_ms_mean": grp["median_latency_ms"].mean(),
            "median_latency_ms_std": grp["median_latency_ms"].std(ddof=1) if len(grp) > 1 else 0.0,
            "peak_vram_mb_mean": grp["peak_vram_mb"].mean(),
            "peak_vram_mb_std": grp["peak_vram_mb"].std(ddof=1) if len(grp) > 1 else 0.0,
        })
    # preserve original model ordering
    order = list(raw_df["model"].drop_duplicates())
    summary_df = pd.DataFrame(rows).set_index("model").loc[order].reset_index()
    return summary_df


def main():
    print(f"PROJECT_ROOT resolved to: {PROJECT_ROOT}")
    print(f"N_TRIALS = {N_TRIALS}, LOAD_ONCE_PER_MODEL = {LOAD_ONCE_PER_MODEL}")
    if not os.path.exists(TEST_CSV):
        raise FileNotFoundError(f"TEST_CSV not found at: {TEST_CSV} -- edit the path at the top of this script.")

    test_df = pd.read_csv(TEST_CSV)
    texts = test_df["review_text"].astype(str).tolist()
    print(f"Loaded {len(texts)} test reviews (using full test set).")

    all_trials = []
    for cfg in MODELS:
        if not os.path.exists(cfg["adapter_dir"]):
            print(f"SKIPPED {cfg['name']} -- adapter dir not found: {cfg['adapter_dir']}")
            continue
        try:
            trials = benchmark_model_repeated(cfg, texts, N_TRIALS)
            all_trials.extend(trials)
        except Exception as e:  # noqa: BLE001
            print(f"ERROR benchmarking {cfg['name']}: {e!r}")

    gemma_cfg = next((m for m in MODELS if m["name"] == "Gemma 3 270M"), None)
    qwen_cfg = next((m for m in MODELS if m["name"] == "Qwen3-0.6B"), None)
    if gemma_cfg and qwen_cfg and os.path.exists(gemma_cfg["adapter_dir"]) and os.path.exists(qwen_cfg["adapter_dir"]):
        try:
            trials = benchmark_ensemble_repeated(gemma_cfg, qwen_cfg, texts, N_TRIALS)
            all_trials.extend(trials)
        except Exception as e:  # noqa: BLE001
            print(f"ERROR benchmarking ensemble: {e!r}")
    else:
        print("SKIPPED ensemble benchmark -- Gemma and/or Qwen adapter dir not found.")

    os.makedirs(PROJECT_ROOT / "results", exist_ok=True)
    raw_df = pd.DataFrame(all_trials)
    raw_out_path = PROJECT_ROOT / "results" / "efficiency_benchmark_repeated_raw.csv"
    raw_df.to_csv(raw_out_path, index=False)

    summary_df = summarize(raw_df)
    summary_out_path = PROJECT_ROOT / "results" / "efficiency_benchmark_repeated_summary.csv"
    summary_df.to_csv(summary_out_path, index=False)

    print("\n" + "=" * 100)
    print(f"{'Model':<22} {'Mean Lat (ms)':>18} {'Median Lat (ms)':>18} {'Peak VRAM (MB)':>18}")
    print("-" * 100)
    for _, r in summary_df.iterrows():
        print(f"{r['model']:<22} "
              f"{r['mean_latency_ms_mean']:>10.2f} +/- {r['mean_latency_ms_std']:<5.2f} "
              f"{r['median_latency_ms_mean']:>10.2f} +/- {r['median_latency_ms_std']:<5.2f} "
              f"{r['peak_vram_mb_mean']:>10.1f} +/- {r['peak_vram_mb_std']:<5.1f}")
    print("=" * 100)
    print(f"\nRaw per-trial results:     {raw_out_path}")
    print(f"Summary (mean +/- std):    {summary_out_path}")
    print("\nSend BOTH csv files back -- the summary file's numbers go directly")
    print("into the paper's efficiency table; the raw file is kept for full")
    print("transparency/reproducibility if a reviewer asks to see per-trial data.")


if __name__ == "__main__":
    main()
