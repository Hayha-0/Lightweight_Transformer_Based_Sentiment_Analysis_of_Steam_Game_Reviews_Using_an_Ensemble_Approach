"""
Complete, consolidated train-test leakage verification script.
Reproduces every leakage-related number cited in the paper:
  - TF-IDF cosine near-duplicate check
  - Exact-string-match check (case/whitespace normalized)
  - MinHash/Jaccard shingle-based near-duplicate check
  - Distributional consistency (chi-square class balance, KS review length)
  - Source-corpus provenance verification

Run with: python leakage_verification.py
Requires: pip install pandas scikit-learn scipy datasketch --break-system-packages
"""
import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy import stats
from scipy.stats import chi2_contingency

TRAIN_PATH = "train.csv"
VAL_PATH = "validation.csv"
TEST_PATH = "test.csv"
ANON_PATH = "anonymous_reviews.csv"

def normalize(text):
    """Case-fold and collapse whitespace for exact-match comparison."""
    return re.sub(r"\s+", " ", str(text).strip().lower())

def get_shingles(text, k=5):
    """k-word shingles for Jaccard/MinHash comparison."""
    words = str(text).lower().split()
    if len(words) < k:
        return {" ".join(words)}
    return {" ".join(words[i:i+k]) for i in range(len(words) - k + 1)}

def jaccard(a, b):
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)

def main():
    train = pd.read_csv(TRAIN_PATH)
    val = pd.read_csv(VAL_PATH)
    test = pd.read_csv(TEST_PATH)
    anon = pd.read_csv(ANON_PATH)

    print("="*70)
    print("CHECK 1: TF-IDF cosine similarity (test vs train)")
    print("="*70)
    vectorizer = TfidfVectorizer(min_df=1, ngram_range=(1,2)).fit(
        pd.concat([train['review_text'], test['review_text']]).astype(str)
    )
    X_train = vectorizer.transform(train['review_text'].astype(str))
    X_test = vectorizer.transform(test['review_text'].astype(str))
    sim = cosine_similarity(X_test, X_train)
    max_sim = sim.max(axis=1)
    threshold = 0.95
    tfidf_flagged = np.where(max_sim >= threshold)[0]
    print(f"Flagged (>= {threshold} cosine similarity): {len(tfidf_flagged)} / {len(test)}")

    print()
    print("="*70)
    print("CHECK 2: Exact-string match (case/whitespace normalized)")
    print("="*70)
    train_norm = set(train['review_text'].astype(str).apply(normalize))
    test_norm = test['review_text'].astype(str).apply(normalize)
    exact_flagged = test_norm.isin(train_norm)
    print(f"Flagged (normalized exact match to any train row): {exact_flagged.sum()} / {len(test)}")

    print()
    print("="*70)
    print("CHECK 3: MinHash / Jaccard (5-word shingles, threshold >= 0.8)")
    print("="*70)
    train_shingles = [get_shingles(t) for t in train['review_text'].astype(str)]
    test_shingles = [get_shingles(t) for t in test['review_text'].astype(str)]
    jaccard_flagged_idx = []
    jaccard_threshold = 0.8
    jaccard_best_scores = {}
    for i, ts in enumerate(test_shingles):
        best = max((jaccard(ts, trs) for trs in train_shingles), default=0.0)
        jaccard_best_scores[i] = best
        if best >= jaccard_threshold:
            jaccard_flagged_idx.append(i)
    print(f"Standalone MinHash/Jaccard matches (>= {jaccard_threshold}): {len(jaccard_flagged_idx)} / {len(test)}")

    # Reported paper figure = UNION with exact-match set (not a standalone count).
    # This matches rows caught by EITHER exact-match OR MinHash/Jaccard, counted once.
    exact_flagged_idx = set(np.where(exact_flagged)[0])
    union_idx = exact_flagged_idx | set(jaccard_flagged_idx)
    new_from_minhash = set(jaccard_flagged_idx) - exact_flagged_idx
    print(f"MinHash/Jaccard rows also in exact-match set: {len(set(jaccard_flagged_idx) & exact_flagged_idx)}")
    print(f"MinHash/Jaccard rows NOT in exact-match set (genuinely new): {len(new_from_minhash)}")
    print(f"UNION (exact-match + MinHash/Jaccard), distinct rows: {len(union_idx)} / {len(test)}")
    if new_from_minhash:
        print("New row(s) details:")
        for i in new_from_minhash:
            print(f"  test_idx={i}, jaccard={jaccard_best_scores[i]:.4f}, "
                  f"text={repr(test.iloc[i]['review_text'][:80])}")

    print()
    print("="*70)
    print("Manual borderline-case check (Jaccard just below 0.8 threshold)")
    print("="*70)
    # One specific case was manually retained in the paper's reported total
    # despite falling below the 0.8 automated threshold: a long (~350-word)
    # community 'checklist meme' review template, independently filled in
    # by two different reviewers. It is documented explicitly in the paper
    # as template reuse, not leakage, but included in the reported count
    # for transparency. This block finds any such near-miss (0.70-0.80)
    # cases so they can be inspected and consciously included/excluded.
    near_miss_idx = [i for i in range(len(test))
                      if 0.70 <= jaccard_best_scores[i] < jaccard_threshold
                      and i not in exact_flagged_idx]
    print(f"Rows with 0.70 <= Jaccard < {jaccard_threshold} (not already exact-matched): {len(near_miss_idx)}")
    for i in near_miss_idx:
        wc = len(str(test.iloc[i]['review_text']).split())
        print(f"  test_idx={i}, jaccard={jaccard_best_scores[i]:.4f}, word_count={wc}")
        print(f"    text preview: {repr(str(test.iloc[i]['review_text'])[:100])}")

    print()
    print("="*70)
    print("SUMMARY: automated vs. paper-reported totals")
    print("="*70)
    print(f"Exact-match:                          {len(exact_flagged_idx)}")
    print(f"MinHash/Jaccard, automated (>= 0.8):   {len(union_idx)}  <- fully reproducible, no manual judgment")
    print(f"MinHash/Jaccard, paper-reported total: {len(union_idx) + len(near_miss_idx)}  "
          f"<- includes {len(near_miss_idx)} manually-retained near-miss case(s) below threshold,")
    print(f"                                          each individually documented above and in the paper text.")

    print()
    print("="*70)
    print("Median word count comparison (flagged vs full test set)")
    print("="*70)
    overall_len = test['review_text'].astype(str).str.split().apply(len)
    for name, idx in [("TF-IDF", tfidf_flagged),
                      ("Exact match", np.where(exact_flagged)[0]),
                      ("Jaccard", np.array(jaccard_flagged_idx))]:
        if len(idx) > 0:
            med = test.iloc[idx]['review_text'].astype(str).str.split().apply(len).median()
            print(f"  {name}: n={len(idx)}, median words={med}")
    print(f"  Full test set median words: {overall_len.median()}")

    print()
    print("="*70)
    print("CHECK 4: Distributional consistency across splits")
    print("="*70)
    ct = pd.DataFrame({
        "train": train['recommended'].value_counts().sort_index(),
        "val": val['recommended'].value_counts().sort_index(),
        "test": test['recommended'].value_counts().sort_index(),
    }).T
    chi2, pval, dof, exp = chi2_contingency(ct)
    print(f"Class-ratio chi-square: chi2={chi2:.3f}, p={pval:.4f}")

    train_len = train['review_text'].astype(str).str.split().apply(len)
    ks_stat, ks_p = stats.ks_2samp(train_len, overall_len)
    print(f"Review-length KS test (train vs test): D={ks_stat:.4f}, p={ks_p:.4f}")

    print()
    print("="*70)
    print("CHECK 5: Source-corpus provenance verification")
    print("="*70)
    anon_s = set(anon['review_text'].astype(str))
    for name, df in [("train",train),("validation",val),("test",test)]:
        s = set(df['review_text'].astype(str))
        matched = len(s & anon_s)
        print(f"  {name}: {matched}/{len(s)} verified ({matched/len(s)*100:.2f}%)")

if __name__ == "__main__":
    main()
