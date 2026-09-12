import requests
import pandas as pd
import time
from urllib.parse import quote

games = pd.read_csv("data/games.csv")

all_reviews = []

REVIEWS_PER_GAME = 500
MAX_RETRIES = 10

for _, game in games.iterrows():

    app_id = game["app_id"]
    game_name = game["game_name"]

    print(f"\nCollecting {game_name}...")

    cursor = "*"
    collected = 0
    retry_count = 0

    while collected < REVIEWS_PER_GAME:

        encoded_cursor = quote(cursor)

        url = (
            f"https://store.steampowered.com/appreviews/"
            f"{app_id}"
            f"?json=1"
            f"&language=english"
            f"&cursor={encoded_cursor}"
            f"&num_per_page=100"
        )

        try:
            response = requests.get(url, timeout=60)

            if response.status_code != 200:
                print(
                    f"HTTP Error {response.status_code} "
                    f"for {game_name}"
                )
                break

            data = response.json()

            if "reviews" not in data:
                retry_count += 1

                print(
                    f"Invalid response for {game_name} "
                    f"(Retry {retry_count}/{MAX_RETRIES})"
                )

                if retry_count >= MAX_RETRIES:
                    print(f"Skipping {game_name}")
                    break

                time.sleep(10)
                continue

            retry_count = 0

        except Exception as e:

            retry_count += 1

            print(
                f"Error for {game_name}: {e} "
                f"(Retry {retry_count}/{MAX_RETRIES})"
            )

            if retry_count >= MAX_RETRIES:
                print(f"Skipping {game_name}")
                break

            time.sleep(10)
            continue

        reviews = data["reviews"]

        if len(reviews) == 0:
            print(f"No more reviews available for {game_name}")
            break

        for review in reviews:

            all_reviews.append({
                "app_id": app_id,
                "game_name": game_name,
                "review_id": review["recommendationid"],
                "user_id": review["author"]["steamid"],
                "review_text": review["review"],
                "recommended": review["voted_up"],
                "votes_helpful": review["votes_up"],
                "playtime_forever": review["author"]["playtime_forever"],
                "timestamp_created": review["timestamp_created"],
                "comment_count": review.get(
                    "comment_count", 0
                ),
                "weighted_vote_score": review.get(
                    "weighted_vote_score", 0
                ),
                "steam_purchase": review.get(
                    "steam_purchase", False
                ),
                "received_for_free": review.get(
                    "received_for_free", False
                ),
                "written_during_early_access": review.get(
                    "written_during_early_access", False
                )
            })

            collected += 1

            if collected >= REVIEWS_PER_GAME:
                break

        print(f"{game_name}: {collected}")

        pd.DataFrame(all_reviews).to_csv(
            "data/all_reviews_backup.csv",
            index=False
        )

        cursor = data["cursor"]

        time.sleep(10)

df = pd.DataFrame(all_reviews)

print("\nTotal Reviews Collected:", len(df))

df.to_csv("data/all_reviews.csv", index=False)

print("Saved to data/all_reviews.csv")