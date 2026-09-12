import pandas as pd

df = pd.read_csv("data/all_reviews.csv")

print("Shape:")
print(df.shape)

print("\nGames:")
print(df["game_name"].value_counts())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nUnique Users:")
print(df["user_id"].nunique())

print("\nRecommended Distribution:")
print(df["recommended"].value_counts())