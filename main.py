import csv
import math
import random

# 1. LOAD SONG DATA

def load_songs(filename):
    songs = []
    with open(filename, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            song = {
                'id': row['id'],
                'name': row.get('name', 'Unknown'),
                'artists': row.get('artists', 'Unknown'),
                'valence': float(row.get('valence', 0)),
                'energy': float(row.get('energy', 0)),
                'danceability': float(row.get('danceability', 0)),
                'acousticness': float(row.get('acousticness', 0)),
                'instrumentalness': float(row.get('instrumentalness', 0)),
                'liveness': float(row.get('liveness', 0)),
                'loudness': float(row.get('loudness', 0)),
                'speechiness': float(row.get('speechiness', 0))
            }
            songs.append(song)
    return songs

# 2. NORMALIZE FEATURES

def min_max_normalize_songs(songs, feature_names):
    mins = {}
    maxs = {}

    for feature in feature_names:
        values = [song[feature] for song in songs]
        mins[feature] = min(values)
        maxs[feature] = max(values)

    for song in songs:
        for feature in feature_names:
            min_val = mins[feature]
            max_val = maxs[feature]
            if max_val - min_val == 0:
                song[feature + '_norm'] = 0.0
            else:
                song[feature + '_norm'] = (song[feature] - min_val) / (max_val - min_val)

    return songs

# 3. CONTENT-BASED FILTERING
# Andrew Ng style: score = theta^T x

FEATURES = [
    'valence_norm',
    'energy_norm',
    'danceability_norm',
    'acousticness_norm',
    'instrumentalness_norm',
    'liveness_norm',
    'loudness_norm',
    'speechiness_norm'
]

def get_song_feature_vector(song):
    return [song[f] for f in FEATURES]

def dot_product(a, b):
    total = 0.0
    for i in range(len(a)):
        total += a[i] * b[i]
    return total

def average_vectors(vectors):
    if not vectors:
        return [0.0] * len(FEATURES)

    avg = [0.0] * len(vectors[0])
    for vec in vectors:
        for i in range(len(vec)):
            avg[i] += vec[i]

    for i in range(len(avg)):
        avg[i] /= len(vectors)

    return avg

def build_user_profile_from_liked_songs(songs, liked_song_ids):
    vectors = []
    for song in songs:
        if song['id'] in liked_song_ids:
            vectors.append(get_song_feature_vector(song))
    return average_vectors(vectors)

def mood_vector(mood_name):
    # These are manually designed starting vectors
    moods = {
        'happy':     [0.90, 0.80, 0.80, 0.20, 0.10, 0.30, 0.70, 0.40],
        'sad':       [0.20, 0.20, 0.30, 0.70, 0.30, 0.20, 0.20, 0.30],
        'calm':      [0.50, 0.30, 0.40, 0.80, 0.40, 0.20, 0.40, 0.20],
        'energetic': [0.70, 0.95, 0.85, 0.10, 0.10, 0.50, 0.90, 0.50],
        'focus':     [0.40, 0.35, 0.30, 0.60, 0.70, 0.15, 0.40, 0.10]
    }
    return moods.get(mood_name.lower(), [0.5] * len(FEATURES))

def combine_vectors(v1, v2, alpha=0.5):
    combined = []
    for i in range(len(v1)):
        combined.append(alpha * v1[i] + (1 - alpha) * v2[i])
    return combined

def content_score(user_vector, song):
    song_vector = get_song_feature_vector(song)
    return dot_product(user_vector, song_vector)

def recommend_content_based(songs, user_vector, top_n=10, exclude_song_ids=None):
    if exclude_song_ids is None:
        exclude_song_ids = set()

    scored = []
    for song in songs:
        if song['id'] in exclude_song_ids:
            continue
        score = content_score(user_vector, song)
        scored.append((score, song))

    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[:top_n]

# 4. LOAD RATINGS

def load_ratings(filename):
    ratings = []
    with open(filename, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            ratings.append({
                'user_id': row['user_id'],
                'song_id': row['song_id'],
                'rating': float(row['rating'])
            })
    return ratings

# 5. BUILD INDEX MAPS

def build_index_maps(songs, ratings):
    song_ids = [song['id'] for song in songs]

    user_set = set()
    rated_song_set = set()
    for r in ratings:
        user_set.add(r['user_id'])
        rated_song_set.add(r['song_id'])

    # Keep only songs that exist in songs.csv
    filtered_song_ids = []
    for sid in song_ids:
        if sid in rated_song_set:
            filtered_song_ids.append(sid)

    user_list = sorted(list(user_set))
    song_list = filtered_song_ids

    user_to_index = {}
    song_to_index = {}

    for i, uid in enumerate(user_list):
        user_to_index[uid] = i

    for j, sid in enumerate(song_list):
        song_to_index[sid] = j

    return user_list, song_list, user_to_index, song_to_index

# 6. BUILD Y AND R MATRICES
# Y[i][j] = rating by user j on song i
# R[i][j] = 1 if rating exists, else 0

def build_rating_matrices(ratings, user_to_index, song_to_index):
    num_songs = len(song_to_index)
    num_users = len(user_to_index)

    Y = []
    R = []

    for _ in range(num_songs):
        Y.append([0.0] * num_users)
        R.append([0] * num_users)

    for entry in ratings:
        uid = entry['user_id']
        sid = entry['song_id']
        rating = entry['rating']

        if uid in user_to_index and sid in song_to_index:
            i = song_to_index[sid]
            j = user_to_index[uid]
            Y[i][j] = rating
            R[i][j] = 1

    return Y, R

# 7. COLLABORATIVE FILTERING FROM SCRATCH
# Andrew Ng style matrix factorization
#
# Prediction:
#   y_hat(i,j) = X[i] dot Theta[j]

# Cost:
#   1/2 * sum(R[i][j] * (X[i] dot Theta[j] - Y[i][j])^2)
#   + lambda/2 * sum(Theta^2) + lambda/2 * sum(X^2)


def initialize_matrix(rows, cols):
    matrix = []
    for _ in range(rows):
        row = []
        for _ in range(cols):
            row.append(random.uniform(-0.1, 0.1))
        matrix.append(row)
    return matrix

def collaborative_filtering_train(Y, R, num_features=5, alpha=0.005, lambda_reg=0.1, iterations=500):
    num_songs = len(Y)
    num_users = len(Y[0]) if num_songs > 0 else 0

    X = initialize_matrix(num_songs, num_features)      # song features
    Theta = initialize_matrix(num_users, num_features)  # user preferences

    for it in range(iterations):
        X_grad = [[0.0] * num_features for _ in range(num_songs)]
        Theta_grad = [[0.0] * num_features for _ in range(num_users)]
        cost = 0.0

        for i in range(num_songs):
            for j in range(num_users):
                if R[i][j] == 1:
                    pred = dot_product(X[i], Theta[j])
                    error = pred - Y[i][j]
                    cost += 0.5 * error * error

                    for k in range(num_features):
                        X_grad[i][k] += error * Theta[j][k]
                        Theta_grad[j][k] += error * X[i][k]

        # Regularization
        for i in range(num_songs):
            for k in range(num_features):
                cost += (lambda_reg / 2.0) * (X[i][k] ** 2)
                X_grad[i][k] += lambda_reg * X[i][k]

        for j in range(num_users):
            for k in range(num_features):
                cost += (lambda_reg / 2.0) * (Theta[j][k] ** 2)
                Theta_grad[j][k] += lambda_reg * Theta[j][k]

        # Gradient descent
        for i in range(num_songs):
            for k in range(num_features):
                X[i][k] -= alpha * X_grad[i][k]

        for j in range(num_users):
            for k in range(num_features):
                Theta[j][k] -= alpha * Theta_grad[j][k]

        if it % 50 == 0 or it == iterations - 1:
            print(f"Iteration {it}: cost = {cost:.4f}")

    return X, Theta

def predict_rating(X, Theta, song_index, user_index):
    return dot_product(X[song_index], Theta[user_index])

def recommend_collaborative_for_user(user_id, user_to_index, song_list, X, Theta, Y, R, top_n=10):
    if user_id not in user_to_index:
        return []

    user_index = user_to_index[user_id]
    recommendations = []

    for song_index, song_id in enumerate(song_list):
        # recommend only unrated songs
        if R[song_index][user_index] == 0:
            pred = predict_rating(X, Theta, song_index, user_index)
            recommendations.append((pred, song_id))

    recommendations.sort(key=lambda x: x[0], reverse=True)
    return recommendations[:top_n]

# 8. HYBRID SYSTEM
# final_score = beta * content_score + (1-beta) * collaborative_score

def build_song_lookup(songs):
    lookup = {}
    for song in songs:
        lookup[song['id']] = song
    return lookup

def recommend_hybrid(
    songs,
    user_vector,
    user_id,
    user_to_index,
    song_to_index,
    X,
    Theta,
    Y,
    R,
    beta=0.5,
    top_n=10,
    exclude_song_ids=None
):
    if exclude_song_ids is None:
        exclude_song_ids = set()

    song_lookup = build_song_lookup(songs)
    results = []

    for song in songs:
        sid = song['id']

        if sid in exclude_song_ids:
            continue

        c_score = content_score(user_vector, song)

        cf_score = 0.0
        if user_id in user_to_index and sid in song_to_index:
            ui = user_to_index[user_id]
            si = song_to_index[sid]
            if R[si][ui] == 0:
                cf_score = predict_rating(X, Theta, si, ui)
            else:
                continue

        final_score = beta * c_score + (1 - beta) * cf_score
        results.append((final_score, c_score, cf_score, song))

    results.sort(key=lambda x: x[0], reverse=True)
    return results[:top_n]

# 9. MAIN PROGRAM

def print_song_results(title, recommendations):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)

    for idx, item in enumerate(recommendations, start=1):
        if len(item) == 2:
            score, song = item
            print(f"{idx}. {song['name']} - {song['artists']} | score={score:.4f}")
        else:
            final_score, c_score, cf_score, song = item
            print(
                f"{idx}. {song['name']} - {song['artists']} | "
                f"final={final_score:.4f} | content={c_score:.4f} | collab={cf_score:.4f}"
            )

def main():
    songs_file = "songs.csv"
    ratings_file = "ratings.csv"

    songs = load_songs(songs_file)

    raw_feature_names = [
        'valence',
        'energy',
        'danceability',
        'acousticness',
        'instrumentalness',
        'liveness',
        'loudness',
        'speechiness'
    ]

    songs = min_max_normalize_songs(songs, raw_feature_names)

    ratings = load_ratings(ratings_file)

    # Build content-based user vector
    # Example: mood + liked songs
    chosen_mood = "happy"
    liked_song_ids = set()  # put some known liked song ids here if you have them

    mood_pref = mood_vector(chosen_mood)
    liked_pref = build_user_profile_from_liked_songs(songs, liked_song_ids)

    # If no liked songs are provided, use mood vector only
    if sum(liked_pref) == 0:
        user_vector = mood_pref
    else:
        user_vector = combine_vectors(mood_pref, liked_pref, alpha=0.6)

    # Content-based recommendations
    content_recs = recommend_content_based(
        songs,
        user_vector=user_vector,
        top_n=10,
        exclude_song_ids=liked_song_ids
    )

    print_song_results("CONTENT-BASED RECOMMENDATIONS", content_recs)

    # Collaborative filtering preparation
    user_list, song_list, user_to_index, song_to_index = build_index_maps(songs, ratings)
    Y, R = build_rating_matrices(ratings, user_to_index, song_to_index)

    # Train collaborative filtering model
    X, Theta = collaborative_filtering_train(
        Y, R,
        num_features=8,
        alpha=0.05,
        lambda_reg=0.06,
        iterations=500
    )

    # Example user
    target_user_id = "1"

    # Collaborative recommendations
    collab_recs_raw = recommend_collaborative_for_user(
        user_id=target_user_id,
        user_to_index=user_to_index,
        song_list=song_list,
        X=X,
        Theta=Theta,
        Y=Y,
        R=R,
        top_n=10
    )

    # Convert collaborative song_id results into full song dicts
    song_lookup = build_song_lookup(songs)
    collab_recs = []
    for pred, sid in collab_recs_raw:
        if sid in song_lookup:
            collab_recs.append((pred, song_lookup[sid]))

    print_song_results("COLLABORATIVE RECOMMENDATIONS", collab_recs)

    # Hybrid recommendations
    hybrid_recs = recommend_hybrid(
        songs=songs,
        user_vector=user_vector,
        user_id=target_user_id,
        user_to_index=user_to_index,
        song_to_index=song_to_index,
        X=X,
        Theta=Theta,
        Y=Y,
        R=R,
        beta=0.5,
        top_n=10,
        exclude_song_ids=liked_song_ids
    )

    print_song_results("HYBRID RECOMMENDATIONS", hybrid_recs)


if __name__ == "__main__":
    main()