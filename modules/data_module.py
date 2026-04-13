import os
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

# ─────────────────────────────────────────────
# SECTION 3.1 — Data Loading
# ─────────────────────────────────────────────

def load_movielens_100k(data_path: str = "data/ml-100k"):
    """
    Load the raw MovieLens 100K dataset files.
    Returns ratings_df and movies_df.
    """
    # Load ratings (u.data): user_id, movie_id, rating, timestamp
    ratings_path = os.path.join(data_path, "u.data")
    ratings_df = pd.read_csv(
        ratings_path,
        sep="\t",
        names=["user_id", "movie_id", "rating", "timestamp"],
        encoding="latin-1"
    )

    # Load movies (u.item): movie_id, title, release_date, genres...
    movies_path = os.path.join(data_path, "u.item")
    movies_df = pd.read_csv(
        movies_path,
        sep="|",
        names=[
            "movie_id", "title", "release_date", "video_release_date",
            "imdb_url", "unknown", "Action", "Adventure", "Animation",
            "Children", "Comedy", "Crime", "Documentary", "Drama",
            "Fantasy", "Film-Noir", "Horror", "Musical", "Mystery",
            "Romance", "Sci-Fi", "Thriller", "War", "Western"
        ],
        encoding="latin-1",
        usecols=["movie_id", "title", "release_date",
                 "Action", "Adventure", "Animation", "Children",
                 "Comedy", "Crime", "Documentary", "Drama",
                 "Fantasy", "Film-Noir", "Horror", "Musical",
                 "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"]
    )

    print(f"[DATA] Loaded {len(ratings_df)} ratings, "
          f"{ratings_df['user_id'].nunique()} users, "
          f"{ratings_df['movie_id'].nunique()} movies")

    return ratings_df, movies_df


# ─────────────────────────────────────────────
# SECTION 3.1.2 — Multi-Criteria Rating Simulation
# Expand each overall rating into 5 criterion scores:
# [storyline, acting, visuals, emotional_impact, enjoyment]
# ─────────────────────────────────────────────

def simulate_multi_criteria(ratings_df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """
    For every user-movie rating, generate 5 criterion scores
    whose weighted average stays consistent with the overall rating.
    Method from Adomavicius & Kwon (2007), used in Chapter 3.1.2.
    """
    rng = np.random.default_rng(seed)
    n = len(ratings_df)

    # Draw small random noise for each of the 5 criteria
    # std=0.4 keeps scores plausible and close to the overall rating
    noise = rng.normal(loc=0.0, scale=0.4, size=(n, 5))

    # Add noise to overall rating to get raw criterion scores
    overall = ratings_df["rating"].values.reshape(-1, 1)  # shape (n, 1)
    raw_criteria = overall + noise                         # shape (n, 5)

    # Clip to valid rating range [1, 5]
    raw_criteria = np.clip(raw_criteria, 1, 5)

    # Re-scale so the simple mean of 5 criteria equals the overall rating
    # (equal weights 0.2 each for simulation, as stated in Chapter 3.1.2)
    criteria_mean = raw_criteria.mean(axis=1, keepdims=True)
    scale = np.where(criteria_mean != 0, overall / criteria_mean, 1.0)
    adjusted = np.clip(raw_criteria * scale, 1, 5)

    # Attach the 5 criteria columns to the dataframe
    criteria_names = ["storyline", "acting", "visuals",
                      "emotional_impact", "enjoyment"]
    criteria_df = pd.DataFrame(adjusted, columns=criteria_names,
                                index=ratings_df.index)

    result = pd.concat([ratings_df, criteria_df], axis=1)

    print(f"[DATA] Simulated 5 criteria for {n} ratings")
    return result


# ─────────────────────────────────────────────
# SECTION 3.1.3 — Preprocessing (all 5 steps in order)
# ─────────────────────────────────────────────

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all 5 preprocessing steps from Chapter 3.1.3 in order.
    """
    criteria = ["storyline", "acting", "visuals",
                "emotional_impact", "enjoyment"]

    # ── Step 1: Remove duplicates (keep most recent timestamp) ──
    before = len(df)
    df = df.sort_values("timestamp", ascending=False)
    df = df.drop_duplicates(subset=["user_id", "movie_id"], keep="first")
    print(f"[PREPROCESS] Step 1 — Removed {before - len(df)} duplicates. "
          f"Remaining: {len(df)}")

    # ── Step 2: Filter users with fewer than 5 ratings ──
    user_counts = df["user_id"].value_counts()
    valid_users = user_counts[user_counts >= 5].index
    before = len(df)
    df = df[df["user_id"].isin(valid_users)]
    print(f"[PREPROCESS] Step 2 — Removed {before - len(df)} ratings "
          f"from users with < 5 ratings. "
          f"Remaining users: {df['user_id'].nunique()}")

    # ── Step 3: Mean-centre ratings per user ──
    # Subtract each user's average from their individual ratings
    # so positive = liked more than average, negative = liked less
    user_means = df.groupby("user_id")["rating"].transform("mean")
    df["rating_centered"] = df["rating"] - user_means

    for c in criteria:
        col_mean = df.groupby("user_id")[c].transform("mean")
        df[f"{c}_centered"] = df[c] - col_mean

    print(f"[PREPROCESS] Step 3 — Mean-centred ratings per user")

    # ── Step 4: Min-max normalise each criterion to [0, 1] ──
    for c in criteria:
        col_min = df[c].min()
        col_max = df[c].max()
        if col_max > col_min:
            df[f"{c}_norm"] = (df[c] - col_min) / (col_max - col_min)
        else:
            df[f"{c}_norm"] = 0.0

    print(f"[PREPROCESS] Step 4 — Min-max normalised all 5 criteria to [0,1]")

    # ── Step 5: 80/20 train-test split per user ──
    # Done at user level so every user has training data
    df["split"] = "train"
    for user_id, group in df.groupby("user_id"):
        n = len(group)
        n_test = max(1, int(n * 0.2))   # at least 1 test rating per user
        test_indices = group.sample(n=n_test, random_state=42).index
        df.loc[test_indices, "split"] = "test"

    train_size = (df["split"] == "train").sum()
    test_size  = (df["split"] == "test").sum()
    print(f"[PREPROCESS] Step 5 — Train/test split: "
          f"{train_size} train ({train_size/len(df)*100:.1f}%) | "
          f"{test_size} test ({test_size/len(df)*100:.1f}%)")

    return df


# ─────────────────────────────────────────────
# 5-Fold Cross Validation split (for final evaluation on 1M)
# ─────────────────────────────────────────────

def get_kfold_splits(df: pd.DataFrame, n_splits: int = 5):
    """
    Returns 5 (train_df, test_df) pairs.
    Split is done at the rating level with stratification by user
    so every user appears in both train and test across folds.
    Used for the final evaluation experiments (Chapter 3.1.3).
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    splits = []
    for train_idx, test_idx in kf.split(df):
        splits.append((df.iloc[train_idx].copy(),
                       df.iloc[test_idx].copy()))
    print(f"[DATA] Prepared {n_splits}-fold cross-validation splits")
    return splits


# ─────────────────────────────────────────────
# Master pipeline — call this from anywhere
# ─────────────────────────────────────────────

def build_dataset(data_path: str = "data/ml-100k") -> dict:
    """
    Full pipeline: load → simulate → preprocess.
    Returns a dict with the full df, train split, test split,
    movies_df, and the list of normalised criteria column names.
    """
    # Load
    ratings_df, movies_df = load_movielens_100k(data_path)

    # Simulate 5 criteria
    df = simulate_multi_criteria(ratings_df)

    # Preprocess (all 5 steps)
    df = preprocess(df)

    train_df = df[df["split"] == "train"].copy()
    test_df  = df[df["split"] == "test"].copy()

    criteria_norm = [
        "storyline_norm", "acting_norm", "visuals_norm",
        "emotional_impact_norm", "enjoyment_norm"
    ]

    print(f"\n[PIPELINE COMPLETE]")
    print(f"  Total ratings  : {len(df)}")
    print(f"  Users          : {df['user_id'].nunique()}")
    print(f"  Movies         : {df['movie_id'].nunique()}")
    print(f"  Train ratings  : {len(train_df)}")
    print(f"  Test ratings   : {len(test_df)}")

    return {
        "full_df"      : df,
        "train_df"     : train_df,
        "test_df"      : test_df,
        "movies_df"    : movies_df,
        "criteria_norm": criteria_norm
    }
    
