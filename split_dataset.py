import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("data/clean_reviews.csv")

print("Original Dataset:")
print(df.shape)

# 70% training
train_df, temp_df = train_test_split(
    df,
    test_size=0.30,
    stratify=df["recommended"],
    random_state=42
)

# Remaining 30% split into 15% validation and 15% testing
val_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    stratify=temp_df["recommended"],
    random_state=42
)

print("\nTrain:")
print(train_df.shape)

print("\nValidation:")
print(val_df.shape)

print("\nTest:")
print(test_df.shape)

# Save files
train_df.to_csv(
    "data/train.csv",
    index=False
)

val_df.to_csv(
    "data/validation.csv",
    index=False
)

test_df.to_csv(
    "data/test.csv",
    index=False
)

print("\nFiles saved:")
print("data/train.csv")
print("data/validation.csv")
print("data/test.csv")