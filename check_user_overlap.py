import pandas as pd

df = pd.read_csv("data/all_reviews.csv")

print(
    df.groupby("user_id")["app_id"]
      .nunique()
      .value_counts()
      .sort_index()
)