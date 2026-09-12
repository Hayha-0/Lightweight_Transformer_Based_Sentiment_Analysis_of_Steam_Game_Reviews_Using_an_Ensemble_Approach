import pandas as pd
import re

FILES = {
    'EuroBERT-210M': '/mnt/user-data/uploads/predictions_eurobert.csv',
    'Qwen3-0.6B': '/mnt/user-data/uploads/predictions_qwen.csv',
    'Gemma 3 270M': '/mnt/user-data/uploads/predictions_gemma.csv',
    'MiniCPM4-0.5B': '/mnt/user-data/uploads/predictions_minicpm4.csv',
}

NEGATION_WORDS = ["not", "n't", "never", "no ", "isn't", "wasn't", "aren't", "weren't","can't", "cannot", "won't", "wouldn't", "shouldn't", "don't", "doesn't","didn't", "without", "hardly", "barely", "lacking", "lacks"]
SARCASM_MARKERS = ["lol", "lmao", "yeah right", "totally", "sure...", "great job","wow.", "amazing...", "oh boy", "10/10 would", "*chef's kiss*"]
POS_WORDS = ["great", "love", "amazing", "good", "best", "fun", "awesome","excellent", "enjoy", "recommend", "perfect", "beautiful", "satisfy","satisfying", "solid", "worth"]
NEG_WORDS = ["bad", "terrible", "worst", "hate", "awful", "boring", "broken","buggy", "waste", "disappoint", "trash", "garbage", "unplayable","annoying", "frustrating", "glitch"]

def has_any(text, words):
    t = text.lower()
    return any(w in t for w in words)
def has_excess_punct(text):
    return bool(re.search(r'[!?]{2,}|\.{3,}', text))
def has_allcaps_word(text):
    return bool(re.search(r'\b[A-Z]{3,}\b', text))
def categorize(text):
    t = str(text)
    cats = []
    if has_any(t, NEGATION_WORDS): cats.append("negation")
    if has_any(t, SARCASM_MARKERS) or has_excess_punct(t) or has_allcaps_word(t): cats.append("sarcasm_marker")
    has_pos = has_any(t, POS_WORDS); has_neg = has_any(t, NEG_WORDS)
    if has_pos and has_neg: cats.append("contradiction")
    if len(t.split()) <= 4: cats.append("short")
    if not cats: cats.append("other/ambiguous")
    return cats

summary_rows = []
for name, path in FILES.items():
    df = pd.read_csv(path)
    errors = df[df["true_label"] != df["pred_label"]].copy()
    errors["categories"] = errors["review_text"].apply(categorize)

    cat_counts = {}
    for cats in errors["categories"]:
        for c in cats:
            cat_counts[c] = cat_counts.get(c, 0) + 1

    total_errors = len(errors)
    fp = ((errors["true_label"]==0) & (errors["pred_label"]==1)).sum()  # missed negative -> called positive
    fn = ((errors["true_label"]==1) & (errors["pred_label"]==0)).sum()  # missed positive -> called negative

    print(f"\n{'='*70}\n{name}  ({total_errors} total misclassifications out of 1710)\n{'='*70}")
    print(f"  False positives (true=NEG, pred=POS): {fp}")
    print(f"  False negatives (true=POS, pred=NEG): {fn}")
    print(f"  Error category breakdown (an error can belong to multiple categories):")
    for c, n in sorted(cat_counts.items(), key=lambda x: -x[1]):
        print(f"    {c:18s}: {n:4d} ({n/total_errors*100:.1f}% of errors)")

    summary_rows.append({"model": name, "total_errors": total_errors, "false_pos": fp, "false_neg": fn, **cat_counts})
    errors.to_csv(f"/home/claude/analysis/errors_{name.replace(' ','_').replace('.','')}.csv", index=False)

summary = pd.DataFrame(summary_rows).set_index("model")
summary.to_csv("/home/claude/analysis/error_category_summary.csv")
print(f"\n\nSaved per-model error files and error_category_summary.csv")
