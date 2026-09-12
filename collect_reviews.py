import requests
import pandas as pd
import time

APP_ID = 730  # Counter-Strike 2 for testing
NUM_PAGES = 5  # 5 pages × 20 reviews = about 100 reviews

reviews = []
cursor = "*"

for page in range(NUM_PAGES):
    print(f"Downloading page {page + 1}...")

    url = (
        f"https://store.steampowered.com/appreviews/"
        f"{APP_ID}"
        f"?json=1"
        f"&cursor={cursor}"
        f"&num_per_page=20"
    )

    response = requests.get(url)
    data = response.json()

    for review in data["reviews"]:
        reviews.append({
            "review_id": review["recommendationid"],
            "user_id": review["author"]["steamid"],
            "review_text": review["review"],
            "recommended": review["voted_up"],
            "votes_helpful": review["votes_up"],
            "playtime_forever": review["author"]["playtime_forever"],
            "timestamp_created": review["timestamp_created"]
        })

    cursor = data["cursor"]

    time.sleep(1)

df = pd.DataFrame(reviews)

print(df.head())
print(f"\nTotal Reviews: {len(df)}")

df.to_csv("data/cs2_reviews_test.csv", index=False)

print("\nSaved to data/cs2_reviews_test.csv")