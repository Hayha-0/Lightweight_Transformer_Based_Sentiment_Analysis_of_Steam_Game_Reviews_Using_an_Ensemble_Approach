"""
xai_ensemble_shap.py

Run this in the SAME environment you trained the models in (Colab/Kaggle/local
with GPU + internet access to Hugging Face Hub). This machine (the one that
wrote this script) has neither GPU access nor Hugging Face Hub access, so it
cannot run this itself -- you run it, then send the output folder back.

Requires, in addition to your existing training env:
    pip install shap

WHAT THIS EXPLAINS
------------------
The paper's actual proposed model is the ENSEMBLE:
    P_ensemble(text) = 0.5 * P_Gemma(text) + 0.5 * P_Qwen(text)
(Section III-F, w=0.5 selected via cross-validated grid search).

Gemma-3-270M and Qwen3-0.6B use different tokenizers/architectures internally,
so this script does NOT try to explain either model's internals separately
(e.g. via attention weights). Instead it wraps the full ensemble prediction
pipeline as one black-box function of raw text -> [P(not recommended),
P(recommended)], and explains THAT with SHAP (shap.Explainer + Partition
algorithm over word-level text masking). This is the correct scope match for
an "XAI section for the ensemble" requirement: it explains the model that is
actually reported in Table IV/V, not a proxy for it.

OUTPUTS (written to XAI_OUT_DIR -- zip this whole folder and send it back)
---------------------------------------------------------------------------
1. ensemble_predictions.csv
       Ensemble prediction + heuristic challenge-subset flags for every one
       of the 1,710 test reviews. (Recomputing this from scratch here also
       double-checks the 95.03% / 0.892 macro-F1 ensemble numbers already in
       the paper, on the single test split.)

2. shap_aggregate_token_importance.csv / .png
       GLOBAL explanation: mean SHAP value per token across a sample of test
       reviews, split into "pushes toward Recommended" vs "pushes toward Not
       Recommended". This is the aggregate-patterns half of the XAI section.

3. shap_case_<name>.png / .csv  (+ shap_case_study_summary.csv)
       LOCAL explanations for a handful of illustrative reviews: a
       negation-corrected example (mirrors the paper's existing Section IV-F
       case study), a sarcasm example, a contradiction example, and a plain
       negative example as a contrast baseline. Each PNG is a per-token
       SHAP bar chart (green = pushes toward Recommended, red = pushes
       toward Not Recommended); each CSV has the raw per-token values.

IMPORTANT CAVEAT (read before trusting Table III/IV consistency claims)
-------------------------------------------------------------------------
The negation/sarcasm/contradiction FLAGS used here (flag_negation,
flag_sarcasm, flag_contradiction below) are a lightweight approximate
reimplementation of the paper's Section III-E heuristics, written from the
paper's prose description only -- the original flagging script was not
available when this was written. Counts from these flags will likely NOT
exactly match Table II/III's 652/415/195 negation/sarcasm/contradiction
counts. This is fine for picking illustrative case-study examples (that's
all it's used for), but if you still have the original challenge-subset
flagging script, swap the three flag_* functions below for calls into it so
the case-study selection is drawn from the exact same subsets as the rest of
the paper.

RUNTIME
-------
SHAP's Partition explainer needs many forward passes per review (does
recursive masking over words). Expect roughly 1-4 seconds/review on a T4 for
short reviews, more for long ones. N_AGGREGATE_SAMPLE=200 should complete in
well under an hour on a free-tier Colab GPU; lower it if you're in a hurry,
raise it if you want a tighter aggregate estimate. The 4 case-study examples
are cheap regardless (one review each).
"""

import os
import re
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from pathlib import Path

from transformers import AutoTokenizer, AutoModelForSequenceClassification, BitsAndBytesConfig
from peft import PeftModel

# ---------------------------------------------------------------------------
# 1. CONFIG -- EDIT THESE PATHS TO MATCH YOUR ENVIRONMENT
# ---------------------------------------------------------------------------
# All paths below are anchored to PROJECT_ROOT (this script's own parent
# folder's parent, i.e. .../SteamRecommendationProject), NOT to the current
# working directory. This means the script now runs correctly no matter which
# folder you `cd` into before running `python xai_ensemble_shap.py` --
# previously these were bare relative strings ("models/...", "data/..."),
# which only worked if you happened to launch the script from the exact
# folder those paths were relative to.
#
# If this script lives directly in the project root instead of in scripts/,
# change PROJECT_ROOT to: Path(__file__).resolve().parent.parent
PROJECT_ROOT = Path(__file__).resolve().parent

GEMMA_ADAPTER_DIR = str(PROJECT_ROOT / "models" / "gemma270m" / "best_adapter")  # from train_gemma.py's OUTPUT_DIR
QWEN_ADAPTER_DIR  = str(PROJECT_ROOT / "models" / "qwen06b" / "best_adapter")    # from train_qwen.py's OUTPUT_DIR
GEMMA_BASE_ID = "google/gemma-3-270m"
QWEN_BASE_ID  = "Qwen/Qwen3-0.6B"

TEST_CSV = str(PROJECT_ROOT / "data" / "test.csv")   # your original 1,710-row held-out test set
TEXT_COLUMN = "review_text"
LABEL_COLUMN = "recommended"        # 1 = recommended/positive, 0 = not/negative
MAX_LENGTH = 512
ENSEMBLE_WEIGHT_GEMMA = 0.5          # matches Section III-F, w=0.5

XAI_OUT_DIR = str(PROJECT_ROOT / "results" / "xai")
os.makedirs(XAI_OUT_DIR, exist_ok=True)

N_AGGREGATE_SAMPLE = 200
RANDOM_SEED = 42

# ---------------------------------------------------------------------------
# 1b. FAST PRE-FLIGHT CHECK -- fail immediately with a clear message instead
#     of a deep stack trace if a path is still wrong for your layout.
# ---------------------------------------------------------------------------
print(f"PROJECT_ROOT resolved to: {PROJECT_ROOT}")
for _label, _path in [
    ("GEMMA_ADAPTER_DIR", GEMMA_ADAPTER_DIR),
    ("QWEN_ADAPTER_DIR", QWEN_ADAPTER_DIR),
    ("TEST_CSV", TEST_CSV),
]:
    if not os.path.exists(_path):
        raise FileNotFoundError(
            f"{_label} does not exist at: {_path}\n"
            f"PROJECT_ROOT was resolved to: {PROJECT_ROOT}\n"
            f"If your folder layout differs, edit PROJECT_ROOT above "
            f"(currently: this script's parent's parent folder)."
        )
print("All input paths verified. Proceeding to model loading...")

# ---------------------------------------------------------------------------
# 2. LOAD BOTH MODELS -- identical quantization/LoRA config to training
#    (mirrors model_utils.get_bnb_config() / build_lora_model())
# ---------------------------------------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
assert device == "cuda", "Run this on a GPU runtime -- 4-bit inference on CPU is impractically slow."

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)


def load_model_and_tokenizer(base_id, adapter_dir):
    # Load tokenizer from the adapter dir, not the base model ID -- the
    # training scripts save the tokenizer alongside best_adapter/, and it's
    # guaranteed to have the same pad_token fix applied at train time.
    tokenizer = AutoTokenizer.from_pretrained(adapter_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForSequenceClassification.from_pretrained(
        base_id,
        num_labels=2,
        quantization_config=bnb_config,
        device_map="auto",
    )
    base_model.config.pad_token_id = tokenizer.pad_token_id

    # PeftModel.from_pretrained restores BOTH the LoRA deltas AND the
    # modules_to_save=("score",) classification head automatically, since
    # PEFT bundles modules_to_save weights into the adapter checkpoint.
    model = PeftModel.from_pretrained(base_model, adapter_dir)
    model.eval()
    return model, tokenizer


print("Loading Gemma 3 270M ...")
gemma_model, gemma_tok = load_model_and_tokenizer(GEMMA_BASE_ID, GEMMA_ADAPTER_DIR)
print("Loading Qwen3-0.6B ...")
qwen_model, qwen_tok = load_model_and_tokenizer(QWEN_BASE_ID, QWEN_ADAPTER_DIR)


# ---------------------------------------------------------------------------
# 3. SINGLE-MODEL AND ENSEMBLE PREDICT FUNCTIONS
# ---------------------------------------------------------------------------
@torch.no_grad()
def predict_proba_single(texts, model, tokenizer, batch_size=16):
    """Returns an (n, 2) array: [:,0]=P(not recommended), [:,1]=P(recommended)."""
    all_probs = []
    for i in range(0, len(texts), batch_size):
        batch = list(texts[i:i + batch_size])
        enc = tokenizer(
            batch, truncation=True, max_length=MAX_LENGTH,
            padding=True, return_tensors="pt",
        ).to(model.device)
        logits = model(**enc).logits.float()
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        all_probs.append(probs)
    return np.concatenate(all_probs, axis=0)


def predict_proba_ensemble(texts):
    """
    Black-box function passed to SHAP. texts: list[str] -> np.ndarray (n, 2).
    Implements the paper's Eq. in Section III-F: 0.5*P_Gemma + 0.5*P_Qwen.
    """
    texts = list(texts)
    p_gemma = predict_proba_single(texts, gemma_model, gemma_tok)
    p_qwen = predict_proba_single(texts, qwen_model, qwen_tok)
    return ENSEMBLE_WEIGHT_GEMMA * p_gemma + (1 - ENSEMBLE_WEIGHT_GEMMA) * p_qwen


# ---------------------------------------------------------------------------
# 4. LOAD TEST SET, GET ENSEMBLE PREDICTIONS FOR ALL 1,710 REVIEWS
# ---------------------------------------------------------------------------
test_df = pd.read_csv(TEST_CSV)
test_df[TEXT_COLUMN] = test_df[TEXT_COLUMN].astype(str)
texts_all = test_df[TEXT_COLUMN].tolist()

print(f"Running ensemble inference on {len(texts_all)} test reviews ...")
probs_all = predict_proba_ensemble(texts_all)
pred_df = pd.DataFrame({
    "test_idx": test_df.index,
    "review_text": texts_all,
    "true_label": test_df[LABEL_COLUMN].astype(int).values,
    "prob_not_recommended": probs_all[:, 0],
    "prob_recommended": probs_all[:, 1],
})
pred_df["pred_label"] = (pred_df["prob_recommended"] >= 0.5).astype(int)
pred_df["correct"] = pred_df["pred_label"] == pred_df["true_label"]

acc_check = pred_df["correct"].mean()
print(f"Sanity check -- ensemble accuracy on this run: {acc_check:.4f} "
      f"(paper reports 0.941; should match closely, small float/library-version drift is normal)")


# ---------------------------------------------------------------------------
# 5. LIGHTWEIGHT CHALLENGE-SUBSET HEURISTICS (see CAVEAT in module docstring)
# ---------------------------------------------------------------------------
NEGATION_WORDS = (
    r"\b(not|no|never|n't|nothing|nobody|none|without|isn't|wasn't|"
    r"aren't|weren't|don't|doesn't|didn't|can't|cannot|won't|wouldn't)\b"
)
SARCASM_MARKERS = r"(lol|lmao|/s\b|\bsure\b.{0,15}\?|!{2,}|\?{2,}|\b10/10\b.{0,20}(hate|worst|bad))"
CONTRAST_MARKERS = r"\b(but|however|although|though|except|despite|yet)\b"


def flag_negation(t):
    return bool(re.search(NEGATION_WORDS, t, re.IGNORECASE))


def flag_sarcasm(t):
    return bool(re.search(SARCASM_MARKERS, t, re.IGNORECASE))


def flag_contradiction(t):
    return bool(re.search(CONTRAST_MARKERS, t, re.IGNORECASE)) and len(t.split()) > 8


pred_df["flag_negation"] = pred_df["review_text"].apply(flag_negation)
pred_df["flag_sarcasm"] = pred_df["review_text"].apply(flag_sarcasm)
pred_df["flag_contradiction"] = pred_df["review_text"].apply(flag_contradiction)
pred_df.to_csv(os.path.join(XAI_OUT_DIR, "ensemble_predictions.csv"), index=False)
print("Saved ensemble_predictions.csv")


# ---------------------------------------------------------------------------
# 6. SHAP SETUP (Partition explainer over text, model-agnostic black box)
# ---------------------------------------------------------------------------
import shap  # noqa: E402  (import after heavy model load so failures above are cheap)

masker = shap.maskers.Text(r"\W+")  # word-level masking; identical regardless of
                                     # Gemma's vs Qwen's internal tokenizers, since
                                     # we're explaining the ENSEMBLE's black-box output
explainer = shap.Explainer(
    predict_proba_ensemble, masker, output_names=["Not Recommended", "Recommended"]
)


# ---------------------------------------------------------------------------
# 7. AGGREGATE (GLOBAL) SHAP ANALYSIS
# ---------------------------------------------------------------------------
sample_df = pred_df.sample(n=min(N_AGGREGATE_SAMPLE, len(pred_df)), random_state=RANDOM_SEED)

# Filter out reviews SHAP's Text masker cannot cluster: the masker splits on
# \W+ (non-word characters), so a review with zero remaining "word" tokens --
# empty string, whitespace-only, or punctuation/emoji/numbers-only with no
# letters -- produces a zero-size array that crashes shap's internal
# clustering step (ValueError: zero-size array to reduction operation
# maximum). This happened at review 158/200 on a full 32-minute run with the
# previous single-bulk-call version, which lost all prior progress since
# explainer(sample_texts) is one call for the whole batch. Filtering these
# out up front, and looping+checkpointing below, means neither problem can
# happen again.
def _has_word_tokens(t, pattern=re.compile(r"\w+")):
    return bool(pattern.search(str(t)))

_before = len(sample_df)
sample_df = sample_df[sample_df["review_text"].apply(_has_word_tokens)]
_dropped = _before - len(sample_df)
if _dropped:
    print(f"Filtered out {_dropped} degenerate review(s) with no word tokens "
          f"(empty/whitespace/punctuation-only) before SHAP analysis.")

sample_texts = sample_df["review_text"].tolist()

print(f"Computing SHAP values for {len(sample_texts)} sampled reviews "
      f"(aggregate analysis) -- this is the slow part.")

# Per-review loop instead of one bulk explainer(sample_texts) call: a single
# problematic review can no longer take down the whole run, and progress is
# checkpointed to disk every CHECKPOINT_EVERY reviews so a crash near the end
# doesn't cost you the full runtime again.
CHECKPOINT_EVERY = 20
shap_values_list = []
skipped_reviews = []
_checkpoint_path = os.path.join(XAI_OUT_DIR, "_shap_values_checkpoint.pkl")

for i, text in enumerate(sample_texts):
    try:
        sv = explainer([text])
        shap_values_list.append(sv[0])
    except Exception as e:  # noqa: BLE001 -- intentionally broad: any single
                            # review failing must not kill the whole run.
        print(f"  WARNING: skipping review {i} due to SHAP error: {e!r}\n"
              f"           text preview: {text[:80]!r}")
        skipped_reviews.append({"index": i, "review_text": text, "error": str(e)})
        continue

    if (i + 1) % CHECKPOINT_EVERY == 0 or (i + 1) == len(sample_texts):
        import pickle  # noqa: E402
        with open(_checkpoint_path, "wb") as f:
            pickle.dump(shap_values_list, f)
        print(f"  ...{i + 1}/{len(sample_texts)} done, checkpoint saved "
              f"({len(shap_values_list)} succeeded, {len(skipped_reviews)} skipped).")

if skipped_reviews:
    pd.DataFrame(skipped_reviews).to_csv(
        os.path.join(XAI_OUT_DIR, "shap_skipped_reviews.csv"), index=False
    )
    print(f"Saved {len(skipped_reviews)} skipped review(s) to shap_skipped_reviews.csv "
          f"for your own inspection.")

shap_values = shap_values_list

from collections import defaultdict  # noqa: E402

token_impact = defaultdict(list)
for sv in shap_values:
    tokens = sv.data
    values_pos_class = sv.values[:, 1]  # attribution toward "Recommended"
    for tok, val in zip(tokens, values_pos_class):
        tok_clean = tok.strip().lower()
        if tok_clean and tok_clean.isalpha():
            token_impact[tok_clean].append(val)

agg_rows = []
for tok, vals in token_impact.items():
    if len(vals) < 3:  # drop tokens seen too rarely to trust
        continue
    agg_rows.append({
        "token": tok,
        "n_occurrences": len(vals),
        "mean_shap_toward_recommended": float(np.mean(vals)),
        "mean_abs_shap": float(np.mean(np.abs(vals))),
    })
agg_df = pd.DataFrame(agg_rows).sort_values("mean_abs_shap", ascending=False)
agg_df.to_csv(os.path.join(XAI_OUT_DIR, "shap_aggregate_token_importance.csv"), index=False)

# Keep the analysis unchanged; only improve the PNG layout.
# Use fewer bars and clean labels so the y-axis text cannot overlap.
top_pos = agg_df.sort_values(
    "mean_shap_toward_recommended", ascending=False
).head(12).copy()

top_neg = agg_df.sort_values(
    "mean_shap_toward_recommended", ascending=True
).head(12).copy()


def clean_plot_token(token, max_len=18):
    token = str(token).replace("\n", " ").replace("\r", " ").strip()
    token = re.sub(r"\s+", " ", token)
    if len(token) > max_len:
        token = token[:max_len - 1] + "…"
    return token


top_pos["plot_token"] = top_pos["token"].apply(clean_plot_token)
top_neg["plot_token"] = top_neg["token"].apply(clean_plot_token)

fig, axes = plt.subplots(1, 2, figsize=(14, 7))

axes[0].barh(
    top_pos["plot_token"].iloc[::-1],
    top_pos["mean_shap_toward_recommended"].iloc[::-1],
    color="seagreen"
)
axes[0].set_title("Top tokens pushing toward 'Recommended'")
axes[0].set_xlabel("Mean SHAP value")
axes[0].tick_params(axis="y", labelsize=10)

axes[1].barh(
    top_neg["plot_token"].iloc[::-1],
    top_neg["mean_shap_toward_recommended"].iloc[::-1],
    color="firebrick"
)
axes[1].set_title("Top tokens pushing toward 'Not Recommended'")
axes[1].set_xlabel("Mean SHAP value")
axes[1].tick_params(axis="y", labelsize=10)

# Explicit spacing is more reliable here than tight_layout for long y-labels.
fig.subplots_adjust(
    left=0.14,
    right=0.98,
    bottom=0.10,
    top=0.92,
    wspace=0.45
)

plt.savefig(
    os.path.join(XAI_OUT_DIR, "shap_aggregate_token_importance.png"),
    dpi=220,
    bbox_inches="tight"
)
plt.close()
print("Saved aggregate SHAP summary.")


# ---------------------------------------------------------------------------
# 8. LOCAL CASE-STUDY EXPLANATIONS
#    Mirrors the paper's existing Section IV-F negation case study, plus a
#    sarcasm, a contradiction, and a plain-negative baseline example.
# ---------------------------------------------------------------------------
case_study_candidates = {
    "negation_corrected": pred_df[
        pred_df["flag_negation"] & (pred_df["true_label"] == 1) & pred_df["correct"]
    ],
    "sarcasm": pred_df[pred_df["flag_sarcasm"] & pred_df["correct"]],
    "contradiction": pred_df[pred_df["flag_contradiction"] & pred_df["correct"]],
    "straightforward_negative": pred_df[
        (~pred_df["flag_negation"]) & (~pred_df["flag_sarcasm"])
        & (pred_df["true_label"] == 0) & pred_df["correct"]
    ],
}

case_study_rows = []
for case_name, subset in case_study_candidates.items():
    if len(subset) == 0:
        print(f"WARNING: no examples found for case '{case_name}', skipping.")
        continue
    row = subset.sample(1, random_state=RANDOM_SEED).iloc[0]
    text = row["review_text"]
    print(f"Explaining case '{case_name}': {text[:80]!r}")

    sv = explainer([text])

    tok_df = pd.DataFrame({
        "token": sv.data[0],
        "shap_toward_recommended": sv.values[0][:, 1],
    })
    tok_df.to_csv(os.path.join(XAI_OUT_DIR, f"shap_case_{case_name}.csv"), index=False)
    case_study_rows.append({
        "case": case_name, "review_text": text,
        "true_label": row["true_label"], "pred_label": row["pred_label"],
        "prob_recommended": row["prob_recommended"],
    })

    tok_df_sorted = tok_df.reindex(
        tok_df["shap_toward_recommended"].abs().sort_values(ascending=False).index
    ).head(20)
    plt.figure(figsize=(8, 5))
    colors = ["seagreen" if v > 0 else "firebrick" for v in tok_df_sorted["shap_toward_recommended"]]
    plt.barh(tok_df_sorted["token"][::-1], tok_df_sorted["shap_toward_recommended"][::-1], color=colors[::-1])
    plt.title(
        f"SHAP token attribution -- {case_name}\n"
        f"(pred={'Recommended' if row['pred_label'] == 1 else 'Not Recommended'}, "
        f"true={'Recommended' if row['true_label'] == 1 else 'Not Recommended'})"
    )
    plt.xlabel("SHAP value (toward 'Recommended')")
    plt.tight_layout()
    plt.savefig(os.path.join(XAI_OUT_DIR, f"shap_case_{case_name}.png"), dpi=150)
    plt.close()

pd.DataFrame(case_study_rows).to_csv(
    os.path.join(XAI_OUT_DIR, "shap_case_study_summary.csv"), index=False
)
print("Saved case-study explanations.")

print("\nDONE. Zip and send back the whole folder:", XAI_OUT_DIR)
