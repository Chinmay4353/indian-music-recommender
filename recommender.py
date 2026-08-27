from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "songs_features.csv"
)


# ============================================================
# MUSIC RECOMMENDER
# ============================================================

class MusicRecommender:
    """
    Content-based music recommendation system.

    Uses:
        TF-IDF
        Cosine similarity

    Supports filtering by:
        Language
        Region group
        Genre
    """

    def __init__(self, data_path=DATA_PATH):

        self.data_path = Path(data_path)

        self.df = None
        self.vectorizer = None
        self.tfidf_matrix = None

        self._load_data()
        self._build_model()

    # ========================================================
    # LOAD DATA
    # ========================================================

    def _load_data(self):
        """Load the feature-engineered dataset."""

        if not self.data_path.exists():
            raise FileNotFoundError(
                f"Dataset not found:\n{self.data_path}"
            )

        self.df = pd.read_csv(
            self.data_path,
            encoding="utf-8"
        )

        required_columns = [
            "id",
            "song_name",
            "artist",
            "genre",
            "language",
            "region",
            "region_group",
            "feature_text",
        ]

        missing_columns = [
            column
            for column in required_columns
            if column not in self.df.columns
        ]

        if missing_columns:
            raise ValueError(
                "Dataset is missing required columns: "
                f"{missing_columns}"
            )

        # Make sure feature text never contains NaN
        self.df["feature_text"] = (
            self.df["feature_text"]
            .fillna("")
            .astype(str)
        )

        # Make sure filter columns are strings
        for column in [
            "song_name",
            "artist",
            "genre",
            "language",
            "region",
            "region_group",
        ]:
            self.df[column] = (
                self.df[column]
                .fillna("")
                .astype(str)
                .str.strip()
            )

    # ========================================================
    # BUILD TF-IDF MODEL
    # ========================================================

    def _build_model(self):
        """Build the TF-IDF matrix."""

        print("Building TF-IDF model...")

        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            min_df=1,
        )

        self.tfidf_matrix = (
            self.vectorizer.fit_transform(
                self.df["feature_text"]
            )
        )

        print(
            "TF-IDF model built successfully."
        )

    # ========================================================
    # FIND SONG
    # ========================================================

    def _find_song_indices(self, song_name):
        """
        Find songs matching the requested name.

        Returns all matching indices.
        """

        query = (
            str(song_name)
            .lower()
            .strip()
        )

        matches = self.df[
            self.df["song_name"]
            .str.lower()
            .str.strip()
            == query
        ]

        return matches.index.tolist()

    # ========================================================
    # RECOMMEND
    # ========================================================

    def recommend(
        self,
        song_name,
        top_n=10,
        language=None,
        region_group=None,
        genre=None,
    ):
        """
        Recommend songs similar to a given song.

        Optional filters:
            language
            region_group
            genre
        """

        song_indices = self._find_song_indices(
            song_name
        )

        if not song_indices:
            return pd.DataFrame(
                columns=self.df.columns.tolist()
                + ["similarity"]
            )

        # Use the first matching song
        song_index = song_indices[0]

        # Calculate similarity
        similarity_scores = cosine_similarity(
            self.tfidf_matrix[song_index],
            self.tfidf_matrix,
        ).flatten()

        results = self.df.copy()

        results["similarity"] = (
            similarity_scores
        )

        # ----------------------------------------------------
        # Remove selected song
        # ----------------------------------------------------

        results = results[
            results.index != song_index
        ]

        # ----------------------------------------------------
        # Language filter
        # ----------------------------------------------------

        if language:

            language_query = (
                str(language)
                .lower()
                .strip()
            )

            results = results[
                results["language"]
                .str.lower()
                .str.strip()
                == language_query
            ]

        # ----------------------------------------------------
        # Region filter
        # ----------------------------------------------------

        if region_group:

            region_query = (
                str(region_group)
                .lower()
                .strip()
            )

            results = results[
                results["region_group"]
                .str.lower()
                .str.strip()
                == region_query
            ]

        # ----------------------------------------------------
        # Genre filter
        # ----------------------------------------------------

        if genre:

            genre_query = (
                str(genre)
                .lower()
                .strip()
            )

            results = results[
                results["genre"]
                .str.lower()
                .str.strip()
                == genre_query
            ]

        # ----------------------------------------------------
        # Sort by similarity
        # ----------------------------------------------------

        results = results.sort_values(
            by="similarity",
            ascending=False,
        )

        return results.head(
            top_n
        ).reset_index(drop=True)


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 50)
    print("Indian Music Recommendation Engine")
    print("=" * 50)

    recommender = MusicRecommender()

    print(
        f"\nSongs loaded: "
        f"{len(recommender.df)}"
    )

    print(
        "TF-IDF matrix shape: "
        f"{recommender.tfidf_matrix.shape}"
    )

    # --------------------------------------------------------
    # Test song
    # --------------------------------------------------------

    test_song = (
        recommender.df.iloc[0]["song_name"]
    )

    print(
        f"\nTest song: {test_song}"
    )

    recommendations = recommender.recommend(
        test_song,
        top_n=5,
    )

    print("\nTop recommendations:")

    if recommendations.empty:

        print(
            "No recommendations found."
        )

    else:

        print(
            recommendations[
                [
                    "song_name",
                    "artist",
                    "genre",
                    "language",
                    "region_group",
                    "similarity",
                ]
            ].to_string(index=False)
        )

    print("\n" + "=" * 50)
    print("Recommendation engine test complete.")
    print("=" * 50)