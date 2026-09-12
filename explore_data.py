import pandas as pd

df = pd.read_csv("data/cs2_reviews_test.csv")

print(df.head())
print("\nColumns:")
print(df.columns)

print("\nShape:")
print(df.shape)

print("\nMissing Values:")
print(df.isnull().sum())

print("\nRecommended Distribution:")
print(df["recommended"].value_counts())

print("\nHelpful Vote Statistics:")
print(df["votes_helpful"].describe())

print("\nPlaytime Statistics:")
print(df["playtime_forever"].describe())