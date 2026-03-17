import pandas as pd
import random

songs = pd.read_csv("data.csv")

song_ids = songs["id"].tolist()

ratings = []

for user in range(1,31):

    sampled = random.sample(song_ids,20)

    for s in sampled:

        ratings.append({
            "user_id":user,
            "song_id":s,
            "rating":random.randint(1,5)
        })

ratings_df = pd.DataFrame(ratings)

ratings_df.to_csv("ratings.csv",index=False)