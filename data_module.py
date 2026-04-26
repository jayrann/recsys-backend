"""
data_module.py
==============
Handles all data loading, multi-criteria simulation, and preprocessing.

Pipeline:
  1. Load MovieLens CSV files
  2. Simulate 5 criterion scores from overall ratings
  3. Preprocess: duplicate removal → user filtering → mean-centring
                 → min-max normalisation → 80/20 per-user train-test split
"""

from __future__ import annotations


import os
import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# 1. LOAD MOVIELENS
# ─────────────────────────────────────────────────────────────────────────────

def load_movielens_100k(data_dir: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load MovieLens 100K from the given directory.
    Returns (ratings_df, movies_df).

    ratings_df columns: user_id, movie_id, overall_rating, timestamp
    movies_df  columns: movie_id, title, genres
    """
    ratings_path = os.path.join(data_dir, "u.data")
    items_path   = os.path.join(data_dir, "u.item")

    ratings_df = pd.read_csv(
        ratings_path,
        sep="\t",
        names=["user_id", "movie_id", "overall_rating", "timestamp"],
        encoding="latin-1"
    )

    # Try full 24-column MovieLens format first, fall back to minimal format
    try:
        movies_df = pd.read_csv(
            items_path,
            sep="|",
            names=[
                "movie_id", "title", "release_date", "video_release_date",
                "imdb_url", "unknown", "Action", "Adventure", "Animation",
                "Children", "Comedy", "Crime", "Documentary", "Drama",
                "Fantasy", "Film-Noir", "Horror", "Musical", "Mystery",
                "Romance", "Sci-Fi", "Thriller", "War", "Western"
            ],
            encoding="latin-1",
            usecols=["movie_id", "title"]
        )
    except Exception:
        # Minimal format: movie_id|title
        movies_df = pd.read_csv(
            items_path,
            sep="|",
            names=["movie_id", "title"],
            encoding="latin-1",
            usecols=[0, 1]
        )

    print(f"[data_module] Loaded {len(ratings_df):,} ratings | "
          f"{ratings_df['user_id'].nunique()} users | "
          f"{ratings_df['movie_id'].nunique()} movies")
    return ratings_df, movies_df


def load_movielens_1m(data_dir: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load MovieLens 1M from the given directory.
    Returns (ratings_df, movies_df).
    """
    ratings_path = os.path.join(data_dir, "ratings.dat")
    movies_path  = os.path.join(data_dir, "movies.dat")

    ratings_df = pd.read_csv(
        ratings_path,
        sep="::",
        names=["user_id", "movie_id", "overall_rating", "timestamp"],
        engine="python",
        encoding="latin-1"
    )

    movies_df = pd.read_csv(
        movies_path,
        sep="::",
        names=["movie_id", "title", "genres"],
        engine="python",
        encoding="latin-1"
    )

    print(f"[data_module] Loaded {len(ratings_df):,} ratings | "
          f"{ratings_df['user_id'].nunique()} users | "
          f"{ratings_df['movie_id'].nunique()} movies")
    return ratings_df, movies_df


# ─────────────────────────────────────────────────────────────────────────────
# 2. MULTI-CRITERIA SIMULATION
# ─────────────────────────────────────────────────────────────────────────────

CRITERIA = ["storyline", "acting", "visuals", "emotional_impact", "enjoyment"]
SIGMA    = 0.4
SEED     = 42


def simulate_multi_criteria(ratings_df: pd.DataFrame) -> pd.DataFrame:
    """
    Expand each single overall_rating into 5 criterion scores.

    For each criterion c:
        raw_c = clip(overall_rating + N(0, sigma^2), 1, 5)

    Then rescale so mean(raw_1..5) == overall_rating, and clip again.

    Returns ratings_df with 5 new criterion columns added.
    """
    rng = np.random.default_rng(SEED)
    df  = ratings_df.copy()

    n   = len(df)
    raw = np.zeros((n, 5))

    for c in range(5):
        noise   = rng.normal(0, SIGMA, size=n)
        raw[:, c] = np.clip(df["overall_rating"].values + noise, 1, 5)

    # Rescale so mean equals overall_rating
    row_means = raw.mean(axis=1, keepdims=True)          # shape (n, 1)
    overall   = df["overall_rating"].values.reshape(-1, 1)

    # avoid division by zero for rows where all raw == same value
    safe_mean    = np.where(row_means == 0, 1, row_means)
    scaled       = raw * (overall / safe_mean)
    scaled       = np.clip(scaled, 1, 5)

    for i, cname in enumerate(CRITERIA):
        df[cname] = scaled[:, i].round(4)

    print(f"[data_module] Simulated 5 criterion scores for {n:,} ratings "
          f"(sigma={SIGMA}, seed={SEED})")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 3. PREPROCESSING PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def preprocess(df: pd.DataFrame,
               min_ratings: int = 5,
               test_size: float = 0.20,
               seed: int = SEED) -> dict:
    """
    Five-stage preprocessing pipeline.

    Returns dict with keys:
        full_df   – complete preprocessed DataFrame (normalised, centred)
        train_df  – 80% split per user
        test_df   – 20% split per user
    """

    # ── Stage 1: Duplicate removal ──────────────────────────────────────────
    before = len(df)
    df = (df.sort_values("timestamp", ascending=False)
            .drop_duplicates(subset=["user_id", "movie_id"], keep="first")
            .reset_index(drop=True))
    print(f"[preprocess] Stage 1 — Duplicates removed: {before - len(df)}")

    # ── Stage 2: User filtering ──────────────────────────────────────────────
    rating_counts = df.groupby("user_id")["movie_id"].count()
    valid_users   = rating_counts[rating_counts >= min_ratings].index
    before        = len(df)
    df            = df[df["user_id"].isin(valid_users)].reset_index(drop=True)
    print(f"[preprocess] Stage 2 — Users removed (< {min_ratings} ratings): "
          f"{before - len(df)} rows dropped | "
          f"{df['user_id'].nunique()} users remain")

    # ── Stage 3: Mean-centring ───────────────────────────────────────────────
    for cname in CRITERIA:
        user_means = df.groupby("user_id")[cname].transform("mean")
        df[f"{cname}_centred"] = df[cname] - user_means

    overall_means = df.groupby("user_id")["overall_rating"].transform("mean")
    df["overall_centred"] = df["overall_rating"] - overall_means
    print(f"[preprocess] Stage 3 — Mean-centring applied to all criteria")

    # ── Stage 4: Min-max normalisation ──────────────────────────────────────
    # Normalise original (uncentred) criterion scores to [0, 1]
    # r'(u,i,c) = (r(u,i,c) - 1) / (5 - 1)
    for cname in CRITERIA:
        df[f"{cname}_norm"] = (df[cname] - 1.0) / 4.0

    df["overall_norm"] = (df["overall_rating"] - 1.0) / 4.0
    print(f"[preprocess] Stage 4 — Min-max normalisation applied [0, 1]")

    # ── Stage 5: Per-user 80/20 train-test split ─────────────────────────────
    train_frames = []
    test_frames  = []

    for user_id, group in df.groupby("user_id"):
        n_test = max(1, int(len(group) * test_size))
        test_sample  = group.sample(n=n_test, random_state=seed)
        train_sample = group.drop(test_sample.index)
        test_frames.append(test_sample)
        train_frames.append(train_sample)

    train_df = pd.concat(train_frames).reset_index(drop=True)
    test_df  = pd.concat(test_frames).reset_index(drop=True)

    print(f"[preprocess] Stage 5 — Train/test split: "
          f"{len(train_df):,} train | {len(test_df):,} test")

    return {
        "full_df":  df,
        "train_df": train_df,
        "test_df":  test_df,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 4. MASTER PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def build_dataset(data_dir: str, version: str = "100k") -> dict:
    """
    Full pipeline: load → simulate → preprocess.

    version: "100k" or "1m"

    Returns dict with:
        full_df, train_df, test_df, movies_df
    """
    print(f"\n{'='*60}")
    print(f"[data_module] Building dataset — MovieLens {version.upper()}")
    print(f"{'='*60}")

    if version == "100k":
        ratings_df, movies_df = load_movielens_100k(data_dir)
    elif version == "1m":
        ratings_df, movies_df = load_movielens_1m(data_dir)
    else:
        raise ValueError(f"Unknown version: {version}. Use '100k' or '1m'.")

    # Simulate multi-criteria scores
    df = simulate_multi_criteria(ratings_df)

    # Preprocess
    result = preprocess(df)
    result["movies_df"] = movies_df

    print(f"\n[data_module] Dataset ready.")
    print(f"  Users   : {result['full_df']['user_id'].nunique():,}")
    print(f"  Movies  : {result['full_df']['movie_id'].nunique():,}")
    print(f"  Ratings : {len(result['full_df']):,}")
    print(f"  Train   : {len(result['train_df']):,}")
    print(f"  Test    : {len(result['test_df']):,}")
    print(f"{'='*60}\n")

    return result
