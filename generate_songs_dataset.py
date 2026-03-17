import pandas as pd

# Load original Spotify file
df = pd.read_csv("data.csv")

# Keep only required columns
songs = df[
    [
        "id",
        "name",
        "artists",
        "valence",
        "energy",
        "danceability",
        "acousticness",
        "instrumentalness",
        "liveness",
        "loudness",
        "speechiness",
        "tempo",
        "popularity",
        "year"
    ]
].copy()

# Clean artists column
songs["artists"] = songs["artists"].str.replace("[", "", regex=False)
songs["artists"] = songs["artists"].str.replace("]", "", regex=False)
songs["artists"] = songs["artists"].str.replace("'", "", regex=False)

# Remove rows with missing important values
songs = songs.dropna(
    subset=[
        "id",
        "name",
        "artists",
        "valence",
        "energy",
        "danceability",
        "acousticness",
        "instrumentalness",
        "liveness",
        "loudness",
        "speechiness",
        "tempo"
    ]
)

# Remove duplicate songs by Spotify id
songs = songs.drop_duplicates(subset=["id"])

# Optional: reduce size for faster testing
# songs = songs.sample(5000, random_state=42)

# Save final songs file
songs.to_csv("songs.csv", index=False)

print("songs.csv created successfully")
print(songs.head())
print(songs.shape)