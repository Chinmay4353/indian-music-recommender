from pathlib import Path

import pandas as pd


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "songs.csv"
)

FEATURED_DATA_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "songs_features.csv"
)


# ============================================================
# FEATURE COLUMNS
# ============================================================

TEXT_FEATURE_COLUMNS = [
    "song_name",
    "artist",
    "genre",
    "language",
    "region",
    "region_group",
    "lyrics",
    "history",
]


# ============================================================
# CREATE FEATURE TEXT
# ============================================================

def create_feature_text(df):
    """
    Combine relevant song metadata and text into one
    feature string for TF-IDF.
    """

    result = df.copy()

    # Make sure all expected columns exist
    for column in TEXT_FEATURE_COLUMNS:
        if column not in result.columns:
            result[column] = ""

    # Replace missing values
    for column in TEXT_FEATURE_COLUMNS:
        result[column] = (
            result[column]
            .fillna("")
            .astype(str)
            .str.strip()
        )

    # Combine metadata and text
    result["feature_text"] = (
        "song " + result["song_name"] + " "
        + "artist " + result["artist"] + " "
        + "genre " + result["genre"] + " "
        + "language " + result["language"] + " "
        + "region " + result["region"] + " "
        + "region_group " + result["region_group"] + " "
        + "lyrics " + result["lyrics"] + " "
        + "history " + result["history"]
    )

    return result


# ============================================================
# LOAD DATASET
# ============================================================

def load_dataset():
    """
    Load the processed song dataset.
    """

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Processed dataset not found:\n{DATA_PATH}"
        )

    return pd.read_csv(
        DATA_PATH,
        encoding="utf-8"
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("Loading processed dataset...")

    df = load_dataset()

    print(
        f"Rows loaded: {len(df)}"
    )

    print(
        "\nCreating feature text..."
    )

    df = create_feature_text(df)

    print(
        "Feature text created successfully."
    )

    print(
        "\nSaving feature-engineered dataset..."
    )

    FEATURED_DATA_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(
        FEATURED_DATA_PATH,
        index=False,
        encoding="utf-8-sig"
    )

    print(
        f"Saved to:\n{FEATURED_DATA_PATH}"
    )

    print(
        f"Rows saved: {len(df)}"
    )

    print(
        "\nFeature engineering complete."
    )