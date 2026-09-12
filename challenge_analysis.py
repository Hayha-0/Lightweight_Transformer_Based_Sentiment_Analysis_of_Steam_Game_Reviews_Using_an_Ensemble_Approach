import pandas as pd
import re

FILES = {
    'EuroBERT-210M': '/mnt/user-data/uploads/predictions_eurobert.csv',
    'Qwen3-0.6B': '/mnt/user-data/uploads/predictions_qwen.csv',
    'Gemma 3 270M': '/mnt/user-data/uploads/predictions_gemma.csv',
    'MiniCPM4-0.5B': '/mnt/user-data/uploads/predictions_minicpm4.csv',
}

NEGATION_WORDS = [
    "not", "n't", "never", "no ", "isn't", "wasn't", "aren't", "weren't",
    "can't", "cannot", "won't", "wouldn't", "shouldn't", "don't", "doesn't",
    "didn't", "without", "hardly", "barely", "lacking", "lacks"
]

SARCASM_MARKERS = [
    "lol", "lmao", "yeah right", "totally", "sure...", "great job",
    "wow.", "amazing...", "oh boy", "10/10 would", "*chef's kiss*"
]

POS_WORDS = [
    "great", "love", "amazing", "good", "best", "fun", "awesome",
    "excellent", "enjoy", "recommend", "perfect", "beautiful", "satisfy",
    "satisfying", "solid", "worth"
]
NEG_WORDS = [
    "bad", "terrible", "worst", "hate", "awful", "boring", "broken",
    "buggy", "waste", "disappoint", "trash", "garbage", "unplayable",
    "annoying", "frustrating", "glitch"
]

def has_any(text, words):
    t = text.lower()
    return any(w in t for w in words)

def has_excess_punct(text):
    return bool(re.search(r'[!?]{2,}|\.{3,}', text))

def has_allcaps_word(text):
    return bool(re.search(r'\b[A-Z]{3,}\b', text))

def flag_row(text):
    t = str(text)
    negation = has_any(t, NEGATION_WORDS)
    sarcasm_marker = has_any(t, SARCASM_MARKERS) or has_excess_punct(t) or has_allcaps_word(t)
    has_pos = has_any(t, POS_WORDS)
    has_neg = has_any(t, NEG_WORDS)
    contradiction = has_pos and has_neg
    short = len(t.split()) <= 4
    return pd.Series({
        "negation": negation,
        "sarcasm_marker": sarcasm_marker,
        "contradiction": contradiction,
        "short": short,
    })

# Load all, verify identical review sets, build flags once (they share the same test set)
base = None
model_dfs = {}
for name, path in FILES.items():
    df = pd.read_csv(path)
    df = df.sort_values("review_text").reset_index(drop=True)
    model_dfs[name] = df
    if base is None:
        base = df[["review_text", "true_label"]].copy()

flags = base["review_text"].apply(flag_row)
base = pd.concat([base, flags], axis=1)

print("="*70)
print("CHALLENGE SUBSET SIZES (out of 1710 total test reviews)")
print("="*70)
for col in ["negation", "sarcasm_marker", "contradiction", "short"]:
    n = base[col].sum()
    print(f"  {col:15s}: {n:4d} reviews ({n/len(base)*100:.1f}%)")

any_challenge = base[["negation","sarcasm_marker","contradiction","short"]].any(axis=1)
print(f"  {'ANY challenge':15s}: {any_challenge.sum():4d} reviews ({any_challenge.sum()/len(base)*100:.1f}%)")

print()
print("="*70)
print("PER-MODEL ACCURACY: FULL SET vs CHALLENGE SUBSETS")
print("="*70)

results = []
for name, df in model_dfs.items():
    df = df.sort_values("review_text").reset_index(drop=True)
    merged = df.merge(base[["review_text","negation","sarcasm_marker","contradiction","short"]], on="review_text")
    row = {"model": name}
    row["full_acc"] = (merged["true_label"]==merged["pred_label"]).mean()
    for col in ["negation", "sarcasm_marker", "contradiction", "short"]:
        sub = merged[merged[col]]
        row[f"{col}_acc"] = (sub["true_label"]==sub["pred_label"]).mean() if len(sub) else float("nan")
        # also negative-class recall within subset
        subneg = sub[sub["true_label"]==0]
        row[f"{col}_neg_recall"] = (subneg["true_label"]==subneg["pred_label"]).mean() if len(subneg) else float("nan")
    any_sub = merged[merged[["negation","sarcasm_marker","contradiction","short"]].any(axis=1)]
    row["any_challenge_acc"] = (any_sub["true_label"]==any_sub["pred_label"]).mean()
    results.append(row)

res_df = pd.DataFrame(results).set_index("model")
pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 20)
print(res_df.round(4).to_string())

res_df.to_csv("/home/claude/analysis/challenge_subset_results.csv")
print("\nSaved: challenge_subset_results.csv")
