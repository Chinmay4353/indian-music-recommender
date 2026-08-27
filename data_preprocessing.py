from pathlib import Path
import ast
import re

import numpy as np
import pandas as pd


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

SPOTIFY_PATH = BASE_DIR / "data" / "raw" / "spotify_dataset.csv"
MARATHI_PATH = BASE_DIR / "data" / "raw" / "marathi_folk_songs.csv"
OUTPUT_PATH = BASE_DIR / "data" / "processed" / "songs.csv"


# ============================================================
# FINAL DATASET SCHEMA
# ============================================================

FINAL_COLUMNS = [
    "id",
    "song_name",
    "artist",
    "genre",
    "mood",
    "language",
    "region",
    "album",
    "year",
    "popularity",
    "cover_image",
    "spotify_url",
    "youtube_url",
    "source",
]


# ============================================================
# GENERAL CLEANING FUNCTIONS
# ============================================================

def clean_text(value):
    """
    Clean text while preserving Indian-language characters.
    """

    if pd.isna(value):
        return np.nan

    text = str(value)

    # Replace newlines/tabs/multiple spaces with one space
    text = re.sub(r"\s+", " ", text).strip()

    return text if text else np.nan


def clean_genre(value):
    """
    Normalize genre text and remove unnecessary whitespace.
    """

    cleaned = clean_text(value)

    if pd.isna(cleaned):
        return np.nan

    cleaned = re.sub(
        r"\s+",
        " ",
        cleaned
    ).strip()

    return cleaned.lower()


def clean_artists(value):
    """
    Convert Spotify list-like artist data into readable text.

    Example:

    ['Arijit Singh', 'Shreya Ghoshal']

    becomes:

    Arijit Singh, Shreya Ghoshal
    """

    if pd.isna(value):
        return np.nan

    # Already a Python list
    if isinstance(value, list):

        artists = [
            str(artist).strip()
            for artist in value
            if str(artist).strip()
        ]

        return ", ".join(artists) if artists else np.nan

    text = str(value).strip()

    # Safely parse strings such as:
    # "['Arijit Singh', 'Shreya Ghoshal']"
    if text.startswith("[") and text.endswith("]"):

        try:
            parsed = ast.literal_eval(text)

            if isinstance(parsed, list):

                artists = [
                    str(artist).strip()
                    for artist in parsed
                    if str(artist).strip()
                ]

                return ", ".join(artists) if artists else np.nan

        except (ValueError, SyntaxError):
            pass

    return clean_text(text)


def extract_year(value):
    """
    Extract a four-digit year from a date or year value.

    Examples:

    2024-11-15 -> 2024
    2024       -> 2024
    """

    if pd.isna(value):
        return np.nan

    match = re.search(
        r"\b(19\d{2}|20\d{2})\b",
        str(value)
    )

    if match:
        return int(match.group(1))

    return np.nan

# ============================================================
# SPOTIFY LANGUAGE / REGION MAPPING
# ============================================================

SPOTIFY_LANGUAGE_MAP = {
    "bollywood": "Hindi",
    "tollywood": "Telugu",
    "kollywood": "Tamil",
    "sandalwood": "Kannada",
    "mollywood": "Malayalam",
    "punjabi": "Punjabi",
    "english": "English",
}


SPOTIFY_REGION_MAP = {
    "bollywood": "North India / Hindi Belt",
    "tollywood": "Andhra Pradesh / Telangana",
    "kollywood": "Tamil Nadu",
    "sandalwood": "Karnataka",
    "mollywood": "Kerala",
    "punjabi": "Punjab",
    "english": "International",
}

# ============================================================
# SPOTIFY DATASET PREPROCESSING
# ============================================================

def preprocess_spotify(df):
    """
    Convert Spotify dataset into the common project schema.
    """

    required_columns = [
        "Track ID",
        "Track Name",
        "Artist(s)",
        "Album",
        "Release Date",
        "Cover Image",
        "Popularity",
        "Genre",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "Spotify dataset is missing required columns: "
            f"{missing_columns}"
        )

    spotify = pd.DataFrame()

    spotify["song_name"] = (
        df["Track Name"]
        .apply(clean_text)
    )

    spotify["artist"] = (
        df["Artist(s)"]
        .apply(clean_artists)
    )

    spotify["genre"] = (
        df["Genre"]
        .apply(clean_genre)
    )

    spotify["album"] = (
        df["Album"]
        .apply(clean_text)
    )

    spotify["year"] = (
        df["Release Date"]
        .apply(extract_year)
    )

    spotify["popularity"] = pd.to_numeric(
        df["Popularity"],
        errors="coerce"
    )

    spotify["cover_image"] = (
        df["Cover Image"]
        .apply(clean_text)
    )

    # Build Spotify URL from actual Track ID
    spotify["spotify_url"] = df["Track ID"].apply(
        lambda track_id: (
            f"https://open.spotify.com/track/{str(track_id).strip()}"
            if pd.notna(track_id)
            else np.nan
        )
    )

    # Spotify dataset does not provide these fields
    spotify["mood"] = np.nan

    spotify["language"] = (
        spotify["genre"]
        .map(SPOTIFY_LANGUAGE_MAP)
    )

    spotify["region"] = (
        spotify["genre"]
        .map(SPOTIFY_REGION_MAP)
    )
    spotify["region_group"] = (
        spotify["region"]
    )

    spotify["youtube_url"] = np.nan

    spotify["source"] = "spotify_india"

    return spotify

# ============================================================
# MARATHI REGION GROUPING
# ============================================================

def get_region_group(region):
    """
    Map detailed Marathi regional metadata into a broader
    regional group for filtering and recommendations.
    """

    if pd.isna(region):
        return np.nan

    text = clean_text(region)

    if pd.isna(text):
        return np.nan

    text_lower = text.lower()

    # Konkan
    konkan_keywords = [
        "कोकण",
        "रत्नागिरी",
        "सिंधुदुर्ग",
        "रायगड",
        "पालघर",
        "मुंबई",
        "ठाणे",
        "मालवण",
        "वसई",
        "वरळी",
    ]

    if any(
        keyword in text
        for keyword in konkan_keywords
    ):
        return "Konkan"

    # Western Maharashtra
    western_keywords = [
        "पुणे",
        "सातारा",
        "सांगली",
        "कोल्हापूर",
        "सोलापूर",
        "पंढरपूर",
        "पश्चिम महाराष्ट्र",
        "जेजुरी",
        "देहू",
        "आळंदी",
        "पन्हाळा",
        "सज्जनगड",
    ]

    if any(
        keyword in text
        for keyword in western_keywords
    ):
        return "Western Maharashtra"

    # Marathwada
    marathwada_keywords = [
        "मराठवाडा",
        "औरंगाबाद",
        "बीड",
        "लातूर",
        "उस्मानाबाद",
        "तुळजापूर",
    ]

    if any(
        keyword in text
        for keyword in marathwada_keywords
    ):
        return "Marathwada"

    # Vidarbha
    vidarbha_keywords = [
        "विदर्भ",
        "नागपूर",
        "अकोला",
        "चंद्रपूर",
        "यवतमाळ",
    ]

    if any(
        keyword in text
        for keyword in vidarbha_keywords
    ):
        return "Vidarbha"

    # Northern Maharashtra / Khandesh
    northern_keywords = [
        "जळगाव",
        "नाशिक",
        "धुळे",
        "खानदेश",
    ]

    if any(
        keyword in text
        for keyword in northern_keywords
    ):
        return "Northern Maharashtra"

    return "Other Maharashtra"

# ============================================================
# MARATHI FOLK DATASET PREPROCESSING
# ============================================================

def preprocess_marathi_folk(df):
    """
    Convert Marathi folk dataset into the common project schema.
    """

    required_columns = [
        "Title",
        "Lyrics",
        "Genre",
        "Region",
        "History",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "Marathi folk dataset is missing required columns: "
            f"{missing_columns}"
        )

    folk = pd.DataFrame()

    folk["song_name"] = (
        df["Title"]
        .apply(clean_text)
    )

    # Artist information is not available
    folk["artist"] = "Traditional / Folk"

    folk["genre"] = (
        df["Genre"]
        .apply(clean_genre)
    )

    # Dataset is specifically Marathi folk music
    folk["language"] = "Marathi"

    # Preserve original regional metadata
    folk["region"] = (
        df["Region"]
        .apply(clean_text)
    )

    folk["region_group"] = (
        folk["region"]
        .apply(get_region_group)
    )

    # Preserve lyrics and history for later ML feature engineering
    folk["lyrics"] = (
        df["Lyrics"]
        .apply(clean_text)
    )

    folk["history"] = (
        df["History"]
        .apply(clean_text)
    )

    # Fields not available in this dataset
    folk["album"] = np.nan
    folk["year"] = np.nan
    folk["popularity"] = np.nan
    folk["cover_image"] = np.nan
    folk["spotify_url"] = np.nan
    folk["youtube_url"] = np.nan
    folk["mood"] = np.nan

    folk["source"] = "marathi_folk"

    return folk


# ============================================================
# DATASET MERGING
# ============================================================

def merge_datasets(spotify_df, marathi_df):
    """
    Merge the two standardized datasets.
    """

    combined = pd.concat(
        [spotify_df, marathi_df],
        ignore_index=True,
        sort=False
    )

    return combined


# ============================================================
# EXACT DUPLICATE REMOVAL
# ============================================================

def remove_exact_duplicates(df):
    """
    Remove completely identical rows.
    """

    before = len(df)

    cleaned = (
        df
        .drop_duplicates(keep="first")
        .reset_index(drop=True)
    )

    removed = before - len(cleaned)

    print(
        f"Exact duplicate rows removed: {removed}"
    )

    return cleaned


# ============================================================
# DUPLICATE MATCHING KEYS
# ============================================================

def normalize_matching_text(value):
    """
    Create a normalized value used only for duplicate matching.
    """

    if pd.isna(value):
        return ""

    text = str(value).lower().strip()

    # Remove punctuation
    text = re.sub(
        r"[^\w\s]",
        "",
        text
    )

    # Normalize whitespace
    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    return text


def create_matching_keys(df):
    """
    Create normalized song and artist keys.

    Original values remain unchanged.
    """

    result = df.copy()

    result["song_key"] = (
        result["song_name"]
        .apply(normalize_matching_text)
    )

    result["artist_key"] = (
        result["artist"]
        .apply(normalize_matching_text)
    )

    return result


# ============================================================
# FIND POTENTIAL DUPLICATES
# ============================================================

def find_potential_duplicates(df):
    """
    Find rows that have the same normalized song and artist.

    These are only potential duplicates.
    They are NOT automatically deleted.
    """

    duplicate_mask = df.duplicated(
        subset=[
            "song_key",
            "artist_key",
        ],
        keep=False
    )

    duplicates = (
        df.loc[duplicate_mask]
        .sort_values(
            [
                "song_key",
                "artist_key",
            ]
        )
        .copy()
    )

    return duplicates


# ============================================================
# DUPLICATE REPORT
# ============================================================

def duplicate_report(df):
    """
    Print a report of potential song + artist duplicates.
    """

    duplicates = find_potential_duplicates(df)

    if duplicates.empty:

        print(
            "\nNo potential song + artist duplicates found."
        )

        return

    duplicate_groups = (
        duplicates
        .groupby(
            [
                "song_key",
                "artist_key",
            ]
        )
        .size()
    )

    print(
        "\n========== DUPLICATE REPORT =========="
    )

    print(
        f"Rows involved: {len(duplicates)}"
    )

    print(
        f"Duplicate groups: {len(duplicate_groups)}"
    )

    print(
        "\nTop duplicate groups:"
    )

    print(
        duplicate_groups
        .sort_values(ascending=False)
        .head(20)
        .to_string()
    )

    print(
        "======================================"
    )


# ============================================================
# FINAL COLUMN PREPARATION
# ============================================================

def prepare_final_columns(df):
    """
    Prepare the dataset for the final standardized schema.

    Additional preprocessing columns such as lyrics, history,
    song_key and artist_key are preserved separately for now.
    """

    result = df.copy()

    # Add missing final-schema columns
    for column in FINAL_COLUMNS:

        if column not in result.columns:
            result[column] = np.nan

    return result


# ============================================================
# GENERATE UNIQUE IDS
# ============================================================

def generate_ids(df):
    """
    Generate a new sequential ID for the merged dataset.
    """

    result = df.copy()

    result["id"] = range(
        1,
        len(result) + 1
    )

    # Keep id as the first column
    columns = [
        "id"
    ] + [
        column
        for column in result.columns
        if column != "id"
    ]

    result = result[columns]

    return result

# ============================================================
# COMPLETE PREPROCESSING PIPELINE
# ============================================================

def preprocess_all():
    """
    Run the complete preprocessing pipeline.
    """

    print(
        "========================================"
    )

    print(
        "Indian Music Dataset Preprocessing"
    )

    print(
        "========================================"
    )

    # --------------------------------------------------------
    # Check input files
    # --------------------------------------------------------

    print(
        "\nChecking input files..."
    )

    if not SPOTIFY_PATH.exists():

        raise FileNotFoundError(
            f"Spotify dataset not found:\n{SPOTIFY_PATH}"
        )

    if not MARATHI_PATH.exists():

        raise FileNotFoundError(
            f"Marathi folk dataset not found:\n{MARATHI_PATH}"
        )

    # --------------------------------------------------------
    # Load datasets
    # --------------------------------------------------------

    print(
        "\nLoading datasets..."
    )

    spotify_raw = pd.read_csv(
        SPOTIFY_PATH
    )

    marathi_raw = pd.read_csv(
        MARATHI_PATH,
        encoding="utf-8"
    )

    print(
        f"Spotify rows: {len(spotify_raw)}"
    )

    print(
        f"Marathi folk rows: {len(marathi_raw)}"
    )

    # --------------------------------------------------------
    # Preprocess Spotify
    # --------------------------------------------------------

    print(
        "\nPreprocessing Spotify dataset..."
    )

    spotify = preprocess_spotify(
        spotify_raw
    )

    print(
        f"Processed Spotify rows: {len(spotify)}"
    )

    # --------------------------------------------------------
    # Preprocess Marathi Folk
    # --------------------------------------------------------

    print(
        "\nPreprocessing Marathi folk dataset..."
    )

    marathi = preprocess_marathi_folk(
        marathi_raw
    )

    print(
        f"Processed Marathi folk rows: {len(marathi)}"
    )

    # --------------------------------------------------------
    # Merge
    # --------------------------------------------------------

    print(
        "\nMerging datasets..."
    )

    combined = merge_datasets(
        spotify,
        marathi
    )

    print(
        f"Rows after merge: {len(combined)}"
    )

    # --------------------------------------------------------
    # Remove exact duplicates
    # --------------------------------------------------------

    print(
        "\nRemoving exact duplicates..."
    )

    combined = remove_exact_duplicates(
        combined
    )

    print(
        f"Rows after exact duplicate removal: "
        f"{len(combined)}"
    )

    # --------------------------------------------------------
    # Create matching keys
    # --------------------------------------------------------

    print(
        "\nCreating duplicate matching keys..."
    )

    combined = create_matching_keys(
        combined
    )

    # --------------------------------------------------------
    # Duplicate report
    # --------------------------------------------------------

    duplicate_report(
        combined
    )

    # --------------------------------------------------------
    # Prepare final schema
    # --------------------------------------------------------

    combined = prepare_final_columns(
        combined
    )

    # --------------------------------------------------------
    # Generate new unique IDs
    # --------------------------------------------------------

    combined = generate_ids(
        combined
    )

    # --------------------------------------------------------
    # Save processed dataset
    # --------------------------------------------------------

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    combined.to_csv(
        OUTPUT_PATH,
        index=False,
        encoding="utf-8-sig"
    )

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print(
        "\n========================================"
    )

    print(
        "PREPROCESSING COMPLETE"
    )

    print(
        "========================================"
    )

    print(
        f"Final rows: {len(combined)}"
    )

    print(
        f"Final columns: {len(combined.columns)}"
    )

    print(
        f"Output file:\n{OUTPUT_PATH}"
    )

    print(
        "========================================"
    )

    return combined


# ============================================================
# SCRIPT ENTRY POINT
# ============================================================

if __name__ == "__main__":
    preprocess_all()