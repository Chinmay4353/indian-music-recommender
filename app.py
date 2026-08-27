from flask import Flask, render_template, request, jsonify
from recommender import MusicRecommender

# ============================================================
# FLASK APP
# ============================================================

app = Flask(__name__)


# ============================================================
# LOAD RECOMMENDER
# ============================================================

print("Loading music recommendation engine...")

recommender = MusicRecommender()

print("Music recommendation engine ready.")


# ============================================================
# HELPERS
# ============================================================

SONG_COLUMNS = [
    "id",
    "song_name",
    "artist",
    "genre",
    "language",
    "region",
    "region_group",
    "album",
    "year",
    "popularity",
    "cover_image",
    "spotify_url",
    "youtube_url",
    "source",
]


def clean_dataframe_for_json(df):
    """
    Convert pandas missing values such as NaN into None
    so Flask can safely return valid JSON.
    """

    result = df.copy()

    return result.astype(object).where(
        result.notna(),
        None
    )


def apply_filters(df, language=None, region_group=None, genre=None):
    """
    Apply optional dataset filters.
    """

    result = df.copy()

    if language:
        query = str(language).strip().lower()

        result = result[
            result["language"]
            .fillna("")
            .astype(str)
            .str.strip()
            .str.lower()
            == query
        ]

    if region_group:
        query = str(region_group).strip().lower()

        result = result[
            result["region_group"]
            .fillna("")
            .astype(str)
            .str.strip()
            .str.lower()
            == query
        ]

    if genre:
        query = str(genre).strip().lower()

        result = result[
            result["genre"]
            .fillna("")
            .astype(str)
            .str.strip()
            .str.lower()
            == query
        ]

    return result


# ============================================================
# HOME
# ============================================================

@app.route("/")
def home():
    return render_template("index.html")


# ============================================================
# SONGS API
# ============================================================

@app.route("/api/songs", methods=["GET"])
def get_songs():

    language = request.args.get("language")
    region_group = request.args.get("region_group")
    genre = request.args.get("genre")

    try:
        limit = int(request.args.get("limit", 50))
    except (TypeError, ValueError):
        limit = 50

    limit = max(1, min(limit, 200))

    # --------------------------------------------------------
    # Apply filters
    # --------------------------------------------------------

    df = apply_filters(
        recommender.df,
        language=language,
        region_group=region_group,
        genre=genre,
    )

    # --------------------------------------------------------
    # Select columns
    # --------------------------------------------------------

    result = df[SONG_COLUMNS].head(limit)

    # --------------------------------------------------------
    # IMPORTANT:
    # Convert NaN -> None
    # --------------------------------------------------------

    result = clean_dataframe_for_json(result)

    return jsonify({
        "count": len(result),
        "songs": result.to_dict(
            orient="records"
        )
    })


# ============================================================
# RECOMMENDATION API
# ============================================================

@app.route("/api/recommend", methods=["GET"])
def recommend():

    song_name = request.args.get("song")

    if not song_name or not song_name.strip():
        return jsonify({
            "error": "Missing 'song' parameter."
        }), 400

    language = request.args.get("language")
    region_group = request.args.get("region_group")
    genre = request.args.get("genre")

    try:
        top_n = int(
            request.args.get("limit", 10)
        )
    except (TypeError, ValueError):
        top_n = 10

    top_n = max(
        1,
        min(top_n, 50)
    )

    # --------------------------------------------------------
    # Run recommendation engine
    # --------------------------------------------------------

    result = recommender.recommend(
        song_name=song_name.strip(),
        top_n=top_n,
        language=language,
        region_group=region_group,
        genre=genre,
    )

    # --------------------------------------------------------
    # No results
    # --------------------------------------------------------

    if result.empty:
        return jsonify({
            "song": song_name.strip(),
            "count": 0,
            "recommendations": []
        })

    # --------------------------------------------------------
    # Select API columns
    # --------------------------------------------------------

    columns = SONG_COLUMNS + [
        "similarity"
    ]

    result = result[columns]

    # --------------------------------------------------------
    # IMPORTANT:
    # Convert NaN -> None
    # --------------------------------------------------------

    result = clean_dataframe_for_json(result)

    return jsonify({
        "song": song_name.strip(),
        "count": len(result),
        "recommendations": result.to_dict(
            orient="records"
        )
    })


# ============================================================
# FILTER OPTIONS API
# ============================================================

@app.route("/api/filters", methods=["GET"])
def get_filters():

    df = recommender.df

    languages = sorted(
        df["language"]
        .dropna()
        .astype(str)
        .str.strip()
        .unique()
        .tolist()
    )

    regions = sorted(
        df["region_group"]
        .dropna()
        .astype(str)
        .str.strip()
        .unique()
        .tolist()
    )

    genres = sorted(
        df["genre"]
        .dropna()
        .astype(str)
        .str.strip()
        .unique()
        .tolist()
    )

    return jsonify({
        "languages": languages,
        "regions": regions,
        "genres": genres,
    })


# ============================================================
# ERROR HANDLERS
# ============================================================

@app.errorhandler(404)
def not_found(error):
    return jsonify({
        "error": "Resource not found."
    }), 404


@app.errorhandler(500)
def server_error(error):
    return jsonify({
        "error": "Internal server error."
    }), 500


# ============================================================
# RUN SERVER
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("Indian Music Recommendation System")
    print("=" * 60)
    print("Server: http://127.0.0.1:5000")
    print("=" * 60)

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True,
    )