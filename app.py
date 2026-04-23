import os
import math
import pandas as pd
from flask import Flask, request, jsonify

app = Flask(__name__)

CSV_PATH = "music_features_subset.csv"

if not os.path.exists(CSV_PATH):
    raise FileNotFoundError(
        f"Dataset not found at '{CSV_PATH}'. "
        "Make sure music_features_subset.csv is in the same directory as app.py."
    )

df = pd.read_csv(CSV_PATH)

REQUIRED_COLS = ["track_name", "artists", "valence", "energy", "loudness", "tempo"]
df = df.dropna(subset=REQUIRED_COLS)

EMOTION_PROFILES = {
    "happy": {
        "filters": lambda row: row["valence"] > 0.6 and row["energy"] > 0.6,
        "target":  {"valence": 0.85, "energy": 0.85, "loudness": -4.0, "tempo": 130.0},
    },
    "sad": {
        "filters": lambda row: row["valence"] < 0.4 and row["energy"] < 0.4,
        "target":  {"valence": 0.15, "energy": 0.15, "loudness": -14.0, "tempo": 70.0},
    },
    "angry": {
        "filters": lambda row: row["energy"] > 0.7 and row["loudness"] > -6,
        "target":  {"valence": 0.30, "energy": 0.90, "loudness": -3.0, "tempo": 150.0},
    },
    "neutral": {
        "filters": lambda row: 0.4 <= row["valence"] <= 0.6,
        "target":  {"valence": 0.50, "energy": 0.50, "loudness": -9.0, "tempo": 100.0},
    },
}

FEATURE_RANGES = {
    "valence":  (0.0, 1.0),
    "energy":   (0.0, 1.0),
    "loudness": (-60.0, 0.0),
    "tempo":    (50.0, 220.0),
}


def normalise(value, feature):
    low, high = FEATURE_RANGES[feature]
    return max(0.0, min(1.0, (value - low) / (high - low)))


def similarity_score(row, target):
    features = ["valence", "energy", "loudness", "tempo"]
    squared_diffs = []
    for f in features:
        song_val   = normalise(row[f], f)
        target_val = normalise(target[f], f)
        squared_diffs.append((song_val - target_val) ** 2)
    distance = math.sqrt(sum(squared_diffs) / len(features))
    return round(1 - distance, 4)


def recommend_songs(emotion, n=10, seen_songs=None):
    """
    seen_songs: list of {"track_name": ..., "artists": ...} dicts to exclude.
    """
    profile  = EMOTION_PROFILES[emotion]
    filtered = df[df.apply(profile["filters"], axis=1)].copy()

    if filtered.empty:
        return []

    # Exclude songs the user has already seen
    if seen_songs:
        seen_set = {(s["track_name"], s["artists"]) for s in seen_songs}
        filtered = filtered[
            ~filtered.apply(lambda r: (r["track_name"], r["artists"]) in seen_set, axis=1)
        ]

    if filtered.empty:
        return []

    target = profile["target"]
    filtered["match_score"] = filtered.apply(
        lambda row: similarity_score(row, target), axis=1
    )

    filtered = filtered[filtered["match_score"] > 0.3]

    if filtered.empty:
        return []

    weights     = filtered["match_score"] ** 2
    sample_size = min(n, len(filtered))
    sampled     = filtered.sample(n=sample_size, weights=weights, replace=False)
    sampled     = sampled.sort_values("match_score", ascending=False)

    return sampled[
        ["track_name", "artists", "valence", "energy", "loudness", "tempo", "match_score"]
    ].to_dict(orient="records")


@app.route("/recommend", methods=["GET"])
def recommend():
    emotion = request.args.get("emotion", "neutral").lower().strip()
    n_param = request.args.get("n", 10)

    try:
        n = int(n_param)
        if n < 1 or n > 50:
            raise ValueError
    except ValueError:
        return jsonify({"error": "Parameter 'n' must be an integer between 1 and 50."}), 400

    if emotion not in EMOTION_PROFILES:
        emotion = "neutral"

    # Accept an optional list of seen song names to exclude
    seen_raw   = request.args.get("seen", "")
    seen_songs = []
    if seen_raw:
        import json
        try:
            seen_songs = json.loads(seen_raw)
        except Exception:
            seen_songs = []

    songs = recommend_songs(emotion, n, seen_songs)

    return jsonify({
        "emotion": emotion,
        "count":   len(songs),
        "songs":   songs,
    })


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status":       "ok",
        "dataset_rows": len(df),
        "emotions":     list(EMOTION_PROFILES.keys()),
    })


@app.route("/emotions", methods=["GET"])
def emotions():
    return jsonify({
        emotion: {"target_features": data["target"]}
        for emotion, data in EMOTION_PROFILES.items()
    })


if __name__ == "__main__":
    app.run(debug=True)