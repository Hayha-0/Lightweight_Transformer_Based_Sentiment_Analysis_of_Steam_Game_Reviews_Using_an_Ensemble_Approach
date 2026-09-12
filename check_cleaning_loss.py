import pandas as pd

df = pd.read_csv("data/all_reviews.csv")

print("Original:", len(df))

duplicates = df["review_id"].duplicated().sum()
print("Duplicate review IDs:", duplicates)

missing = df["review_text"].isna().sum()
print("Missing reviews:", missing)

short1 = (df["review_text"].astype(str).str.len() < 1).sum()
short2 = (df["review_text"].astype(str).str.len() < 2).sum()
short3 = (df["review_text"].astype(str).str.len() < 3).sum()

print("Length < 1:", short1)
print("Length < 2:", short2)
print("Length < 3:", short3)