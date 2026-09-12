import pandas as pd
import re

# ==========================
# Load Dataset
# ==========================

df = pd.read_csv("data/all_reviews.csv")

print("Original Shape:", df.shape)

# ==========================
# Drop Identifier Columns
# ==========================

columns_to_drop = [
    "user_id",
    "review_id",
    "timestamp_created"
]

df = df.drop(columns=columns_to_drop, errors="ignore")

print("Removed Identifier Columns.")

# ==========================
# Anonymize Review Text
# ==========================

def anonymize_text(text):

    if pd.isna(text):
        return text

    text = str(text)

    # Email addresses
    text = re.sub(
        r'\b[\w\.-]+@[\w\.-]+\.\w+\b',
        '[EMAIL]',
        text
    )

    # URLs
    text = re.sub(
        r'https?://\S+|www\.\S+',
        '[URL]',
        text
    )

    # Steam profile links
    text = re.sub(
        r'steamcommunity\.com/\S+',
        '[PROFILE]',
        text,
        flags=re.IGNORECASE
    )

    # Discord usernames
    text = re.sub(
        r'\b[\w]{2,32}#[0-9]{4}\b',
        '[DISCORD]',
        text
    )

    return text

df["review_text"] = df["review_text"].apply(anonymize_text)

print("Review text anonymized.")

# ==========================
# Save Dataset
# ==========================

output_path = "data/anonymous_reviews.csv"

df.to_csv(output_path, index=False)

print("\nSaved:", output_path)

print("\nFinal Shape:", df.shape)

print("\nColumns:")
print(df.columns.tolist())