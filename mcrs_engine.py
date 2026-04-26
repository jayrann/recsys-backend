"""
mcrs_engine.py
==============
Multi-Criteria Collaborative Filtering Engine.

Core components:
  - build_user_item_matrices()  – pivot 5 per-criterion rating matrices
  - weighted_similarity()       – weighted Pearson correlation between users
  - find_neighbours()           – top-K positive-similarity neighbours
  - predict_rating()            – similarity-weighted prediction
  - generate_recommendations()  – top-N unrated movies
  - run_mcrs()                  – entry point used by FastAPI
"""

from __future__ import annotations


import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from typing import Optional


CRITERIA      = ["storyline", "acting", "visuals", "emotional_impact", "enjoyment"]
NORM_COLS     = [f"{c}_norm" for c in CRITERIA]
MIN_CO_RATED  = 5       # minimum co-rated movies for valid Pearson computation
DEFAULT_K     = 30      # neighbourhood size
DEFAULT_TOP_N = 10      # recommendation list size


# ─────────────────────────────────────────────────────────────────────────────
# 1. BUILD USER-ITEM MATRICES
# ─────────────────────────────────────────────────────────────────────────────

def build_user_item_matrices(train_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """
    Build one pivot table per criterion: rows=users, cols=movies.
    Missing entries are NaN (unrated).

    Returns dict: {criterion_name: DataFrame(users x movies)}
    """
    matrices = {}
    for cname in CRITERIA:
        col = f"{cname}_norm"
        if col not in train_df.columns:
            col = cname          # fallback to raw if norm not available
        mat = train_df.pivot_table(
            index="user_id",
            columns="movie_id",
            values=col,
            aggfunc="first"
        )
        matrices[cname] = mat

    print(f"[mcrs_engine] Built {len(matrices)} user-item matrices — "
          f"shape {list(matrices.values())[0].shape}")
    return matrices


# ─────────────────────────────────────────────────────────────────────────────
# 2. WEIGHTED PEARSON SIMILARITY
# ─────────────────────────────────────────────────────────────────────────────

def weighted_similarity(user_a: int,
                        user_b: int,
                        matrices: dict[str, pd.DataFrame],
                        weights: np.ndarray) -> float:
    """
    Compute weighted multi-criteria Pearson similarity between user_a and user_b.

    sim(a, b) = Σc [ wc · pearson_c(a, b) ]

    where pearson_c is computed only over movies rated by BOTH users (co-rated set).

    Returns similarity in [-1, 1], or 0.0 if co-rated count < MIN_CO_RATED.
    """
    sim_total = 0.0

    for i, cname in enumerate(CRITERIA):
        mat = matrices[cname]

        # Check both users exist in the matrix
        if user_a not in mat.index or user_b not in mat.index:
            return 0.0

        row_a = mat.loc[user_a]
        row_b = mat.loc[user_b]

        # Co-rated movies: both must have non-NaN values
        co_rated = row_a.index[row_a.notna() & row_b.notna()]

        if len(co_rated) < MIN_CO_RATED:
            return 0.0

        vec_a = row_a[co_rated].values.astype(float)
        vec_b = row_b[co_rated].values.astype(float)

        # Guard: zero-variance arrays produce undefined Pearson
        if vec_a.std() == 0 or vec_b.std() == 0:
            continue

        corr, _ = pearsonr(vec_a, vec_b)

        if np.isnan(corr):
            continue

        sim_total += weights[i] * corr

    return float(np.clip(sim_total, -1.0, 1.0))


# ─────────────────────────────────────────────────────────────────────────────
# 3. FIND NEIGHBOURS
# ─────────────────────────────────────────────────────────────────────────────

def find_neighbours(target_user: int,
                    matrices: dict[str, pd.DataFrame],
                    weights: np.ndarray,
                    k: int = DEFAULT_K) -> list[tuple[int, float]]:
    """
    Find the top-K most similar users to target_user with positive similarity.

    Returns list of (user_id, similarity) sorted descending by similarity.
    """
    all_users = list(matrices[CRITERIA[0]].index)
    similarities = []

    for user in all_users:
        if user == target_user:
            continue
        sim = weighted_similarity(target_user, user, matrices, weights)
        if sim > 0:
            similarities.append((user, sim))

    # Sort descending by similarity, take top K
    similarities.sort(key=lambda x: x[1], reverse=True)
    neighbours = similarities[:k]

    return neighbours


# ─────────────────────────────────────────────────────────────────────────────
# 4. PREDICT UTILITY SCORE
# ─────────────────────────────────────────────────────────────────────────────

def compute_utility(user_id: int,
                    movie_id: int,
                    matrices: dict[str, pd.DataFrame],
                    weights: np.ndarray) -> Optional[float]:
    """
    Compute the aggregated utility score for a user-movie pair.

    U(u, i) = Σc [ wc · r'(u, i, c) ]

    Returns None if user has not rated the movie.
    """
    utility = 0.0
    for i, cname in enumerate(CRITERIA):
        mat = matrices[cname]
        if user_id not in mat.index or movie_id not in mat.columns:
            return None
        val = mat.loc[user_id, movie_id]
        if pd.isna(val):
            return None
        utility += weights[i] * float(val)
    return float(np.clip(utility, 0.0, 1.0))


def predict_rating(target_user: int,
                   movie_id: int,
                   neighbours: list[tuple[int, float]],
                   matrices: dict[str, pd.DataFrame],
                   weights: np.ndarray) -> Optional[float]:
    """
    Predict the utility score for target_user on movie_id using neighbours.

    Û(u, i) = Σ_v [ sim(u,v) · U(v,i) ] / Σ_v |sim(u,v)|

    Returns None if no neighbour has rated this movie.
    """
    numerator   = 0.0
    denominator = 0.0

    for neighbour_id, sim in neighbours:
        u = compute_utility(neighbour_id, movie_id, matrices, weights)
        if u is None:
            continue
        numerator   += sim * u
        denominator += abs(sim)

    if denominator == 0:
        return None

    prediction = numerator / denominator
    return float(np.clip(prediction, 0.0, 1.0))


# ─────────────────────────────────────────────────────────────────────────────
# 5. GENERATE RECOMMENDATIONS
# ─────────────────────────────────────────────────────────────────────────────

def generate_recommendations(target_user: int,
                              neighbours: list[tuple[int, float]],
                              matrices: dict[str, pd.DataFrame],
                              weights: np.ndarray,
                              movies_df: pd.DataFrame,
                              top_n: int = DEFAULT_TOP_N) -> list[dict]:
    """
    Generate top-N recommendations for target_user.

    Excludes movies the target user has already rated.
    Returns list of dicts: {movie_id, title, predicted_score}.
    """
    # Movies already rated by target user
    first_matrix = matrices[CRITERIA[0]]
    if target_user not in first_matrix.index:
        return []

    rated_movies = set(
        first_matrix.loc[target_user].dropna().index.tolist()
    )

    # All movies that at least one neighbour has rated
    neighbour_ids = [uid for uid, _ in neighbours]
    candidate_movies = set()
    for uid in neighbour_ids:
        if uid in first_matrix.index:
            rated = first_matrix.loc[uid].dropna().index.tolist()
            candidate_movies.update(rated)

    # Remove already-rated movies
    candidate_movies -= rated_movies

    # Predict score for each candidate
    predictions = []
    for movie_id in candidate_movies:
        score = predict_rating(target_user, movie_id, neighbours,
                               matrices, weights)
        if score is not None:
            predictions.append((movie_id, score))

    # Sort by predicted score descending
    predictions.sort(key=lambda x: x[1], reverse=True)
    top = predictions[:top_n]

    # Join with movie titles
    movie_title_map = dict(zip(movies_df["movie_id"], movies_df["title"]))

    recs = []
    for movie_id, score in top:
        recs.append({
            "movie_id":        int(movie_id),
            "title":           movie_title_map.get(movie_id, f"Movie {movie_id}"),
            "predicted_score": round(score, 4)
        })

    return recs


# ─────────────────────────────────────────────────────────────────────────────
# 6. RUN MCRS (FastAPI entry point)
# ─────────────────────────────────────────────────────────────────────────────

def run_mcrs(target_user: int,
             weights: np.ndarray,
             matrices: dict[str, pd.DataFrame],
             movies_df: pd.DataFrame,
             k: int = DEFAULT_K,
             top_n: int = DEFAULT_TOP_N) -> dict:
    """
    Full MCRS pipeline for a single user.

    1. Find K nearest neighbours using weighted Pearson similarity
    2. Predict utility scores for unrated movies
    3. Return top-N recommendation list

    Returns:
        {
          "user_id": int,
          "neighbours_found": int,
          "recommendations": [{"movie_id", "title", "predicted_score"}, ...]
        }
    """
    neighbours = find_neighbours(target_user, matrices, weights, k)

    if not neighbours:
        print(f"[mcrs_engine] No valid neighbours found for user {target_user}")
        return {
            "user_id":          target_user,
            "neighbours_found": 0,
            "recommendations":  []
        }

    recs = generate_recommendations(
        target_user, neighbours, matrices, weights, movies_df, top_n
    )

    print(f"[mcrs_engine] User {target_user} | "
          f"{len(neighbours)} neighbours | "
          f"{len(recs)} recommendations")

    return {
        "user_id":          target_user,
        "neighbours_found": len(neighbours),
        "recommendations":  recs
    }
