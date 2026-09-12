import pandas as pd

df = pd.read_csv("data/anonymous_reviews.csv")

print("Before:", df.shape)

# Remove duplicate reviews based on review text
df = df.drop_duplicates(subset="review_text")

# Remove missing review text
df = df.dropna(subset=["review_text"])

# Ensure review text is string
df["review_text"] = df["review_text"].astype(str)

# Remove extremely short reviews
df = df[df["review_text"].str.len() >= 2]

print("After:", df.shape)

df.to_csv(
    "data/clean_reviews.csv",
    index=False
)

print("Saved clean_reviews.csv")