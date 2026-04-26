import numpy as np
import pandas as pd
from scipy.stats import pearsonr

# ─────────────────────────────────────────────
# SECTION 3.5.1 — Multi-Criteria Rating Representation
# SECTION 3.5.2 — Computing Similarity Between Users
# SECTION 3.5.3 — Recommendation Generation
# ─────────────────────────────────────────────

CRITERIA = [
    "storyline_norm",
    "acting_norm",
    "visuals_norm",
    "emotional_impact_norm",
    "enjoyment_norm"
]

# ─────────────────────────────────────────────
# Step 1 — Build user-item matrices (one per criterion)
# ─────────────────────────────────────────────

def build_user_item_matrices(train_df: pd.DataFrame) -> dict:
    """
    Build one user-item rating matrix per criterion.
    Rows = users, Columns = movies, Values = normalised criterion score.
    NaN means the user has not rated that movie.
    """
    matrices = {}
    for criterion in CRITERIA:
        matrix = train_df.pivot_table(
            index="user_id",
            columns="movie_id",
            values=criterion,
            aggfunc="mean"
        )
        matrices[criterion] = matrix

    print(f"[MCRS] Built {len(CRITERIA)} user-item matrices")
    print(f"[MCRS] Matrix shape: {list(matrices.values())[0].shape} "
          f"(users × movies)")
    return matrices


# ─────────────────────────────────────────────
# Step 2 — Weighted Pearson Similarity
# sim(u, v) = SUM_j [ wj * pearson_j(u, v) ]
# As defined in Chapter 3.5.2
# ─────────────────────────────────────────────

def pearson_per_criterion(
    user_u: int,
    user_v: int,
    matrices: dict,
    min_common: int = 5
) -> dict:
    """
    Compute Pearson correlation between user_u and user_v
    for each of the 5 criteria separately.
    Only uses movies both users have rated (co-rated movies).
    Returns a dict of {criterion: correlation_value}.
    Returns None if fewer than min_common co-rated movies exist.
    """
    correlations = {}

    for criterion, matrix in matrices.items():
        # Get ratings for both users
        if user_u not in matrix.index or user_v not in matrix.index:
            return None

        ratings_u = matrix.loc[user_u]
        ratings_v = matrix.loc[user_v]

        # Find movies both users have rated
        common = ratings_u.notna() & ratings_v.notna()
        common_movies = common[common].index

        # Need minimum co-rated movies for reliable correlation
        if len(common_movies) < min_common:
            return None

        u_scores = ratings_u[common_movies].values
        v_scores = ratings_v[common_movies].values

        # Pearson correlation — handle edge case of zero variance
        if np.std(u_scores) == 0 or np.std(v_scores) == 0:
            correlations[criterion] = 0.0
        else:
            corr, _ = pearsonr(u_scores, v_scores)
            # Clip to [-1, 1] to handle floating point edge cases
            correlations[criterion] = float(np.clip(corr, -1.0, 1.0))

    return correlations


def weighted_similarity(
    user_u: int,
    user_v: int,
    matrices: dict,
    weights: np.ndarray,
    min_common: int = 5
) -> float:
    """
    Compute the weighted multi-criteria similarity between two users.
    sim(u, v) = SUM_j [ wj * pearson_j(u, v) ]
    weights must be a numpy array of 5 values summing to 1.
    Returns 0.0 if insufficient co-rated movies.
    """
    correlations = pearson_per_criterion(
        user_u, user_v, matrices, min_common
    )

    if correlations is None:
        return 0.0

    sim = sum(
        weights[i] * correlations[criterion]
        for i, criterion in enumerate(CRITERIA)
    )

    return float(np.clip(sim, -1.0, 1.0))


# ─────────────────────────────────────────────
# Step 3 — Find Top-N Neighbours
# Select the 30 most similar users (Chapter 3.4.1)
# ─────────────────────────────────────────────

def find_neighbours(
    target_user: int,
    matrices: dict,
    weights: np.ndarray,
    n_neighbours: int = 30,
    min_common: int = 2
) -> list:
    """
    Find the top-N most similar users to the target user.
    Returns a sorted list of (user_id, similarity_score) tuples,
    highest similarity first.
    """
    all_users = list(matrices[CRITERIA[0]].index)
    similarities = []

    for user in all_users:
        if user == target_user:
            continue

        sim = weighted_similarity(
            target_user, user, matrices, weights, min_common
        )

        # Only include positively correlated neighbours
        if sim > 0:
            similarities.append((user, sim))

    # Sort by similarity descending, take top N
    similarities.sort(key=lambda x: x[1], reverse=True)
    neighbours = similarities[:n_neighbours]

    return neighbours


# ─────────────────────────────────────────────
# Step 4 — Predict Rating for a Single Movie
# Weighted average of neighbour ratings
# ─────────────────────────────────────────────

def predict_rating(
    target_user: int,
    movie_id: int,
    neighbours: list,
    matrices: dict,
    weights: np.ndarray,
    train_df: pd.DataFrame
) -> float:
    """
    Predict the weighted aggregated rating for target_user on movie_id.
    Uses the weighted sum formula from Chapter 3.5.1:
    R(u, i) = w1*r1 + w2*r2 + w3*r3 + w4*r4 + w5*r5
    where each rj is predicted from neighbour ratings on criterion j.
    Returns None if no neighbour has rated this movie.
    """
    # Get mean rating of target user for baseline
    user_ratings = train_df[train_df["user_id"] == target_user]["rating"]
    user_mean = user_ratings.mean() if len(user_ratings) > 0 else 3.0

    criterion_predictions = []

    for i, criterion in enumerate(CRITERIA):
        matrix = matrices[criterion]

        numerator   = 0.0
        denominator = 0.0

        for neighbour_id, sim in neighbours:
            if neighbour_id not in matrix.index:
                continue
            neighbour_rating = matrix.loc[neighbour_id, movie_id] \
                if movie_id in matrix.columns else np.nan

            if pd.isna(neighbour_rating):
                continue

            numerator   += sim * neighbour_rating
            denominator += abs(sim)

        if denominator == 0:
            criterion_predictions.append(None)
        else:
            criterion_predictions.append(numerator / denominator)

    # If no neighbour rated this movie on any criterion, skip
    valid = [p for p in criterion_predictions if p is not None]
    if not valid:
        return None

    # Fill missing criterion predictions with mean of valid ones
    filled = [
        p if p is not None else np.mean(valid)
        for p in criterion_predictions
    ]

    # Weighted aggregation: R(u,i) = w1*r1 + w2*r2 + ... + w5*r5
    predicted = float(np.dot(weights, filled))

    # Scale back to [1, 5] from normalised [0, 1]
    predicted_scaled = 1 + predicted * 4
    return float(np.clip(predicted_scaled, 1.0, 5.0))


# ─────────────────────────────────────────────
# Step 5 — Generate Top-N Recommendations
# Return top 10 unrated movies for the target user
# ─────────────────────────────────────────────

def generate_recommendations(
    target_user: int,
    matrices: dict,
    weights: np.ndarray,
    train_df: pd.DataFrame,
    movies_df: pd.DataFrame,
    n_neighbours: int = 30,
    n_recommendations: int = 10,
    min_common: int = 5
) -> pd.DataFrame:
    """
    Full recommendation pipeline for one user:
    1. Find top-30 neighbours using weighted Pearson similarity
    2. For every unrated movie, predict the score
    3. Return top-10 movies with highest predicted scores
    As described in Chapter 3.4.1 flowchart.
    """
    print(f"[MCRS] Generating recommendations for user {target_user}...")

    # Movies the user has already rated — exclude these
    rated_matrix = matrices[CRITERIA[0]]
    if target_user in rated_matrix.index:
        already_rated = set(
            rated_matrix.loc[target_user].dropna().index.tolist()
        )
    else:
        already_rated = set()

    # All movies in the system
    all_movies = set(rated_matrix.columns.tolist())
    unrated_movies = all_movies - already_rated

    print(f"[MCRS] User {target_user} has rated {len(already_rated)} movies. "
          f"Scoring {len(unrated_movies)} unrated movies...")

    # Find neighbours
    neighbours = find_neighbours(
        target_user, matrices, weights, n_neighbours, min_common
    )
    print(f"[MCRS] Found {len(neighbours)} valid neighbours")

    if not neighbours:
        print(f"[MCRS] No neighbours found for user {target_user}")
        return pd.DataFrame()

    # Predict score for every unrated movie
    predictions = []
    for movie_id in unrated_movies:
        score = predict_rating(
            target_user, movie_id, neighbours,
            matrices, weights, train_df
        )
        if score is not None:
            predictions.append({
                "movie_id"       : movie_id,
                "predicted_score": score
            })

    if not predictions:
        print(f"[MCRS] Could not predict scores for user {target_user}")
        return pd.DataFrame()

    # Sort by predicted score, take top N
    pred_df = pd.DataFrame(predictions)
    pred_df = pred_df.sort_values(
        "predicted_score", ascending=False
    ).head(n_recommendations)

    # Merge with movie titles
    result = pred_df.merge(
        movies_df[["movie_id", "title"]],
        on="movie_id",
        how="left"
    )

    result["predicted_score"] = result["predicted_score"].round(2)
    result = result.reset_index(drop=True)
    result.index += 1  # rank starts from 1

    print(f"[MCRS] Top {n_recommendations} recommendations generated")
    return result


# ─────────────────────────────────────────────
# Master function — call this from FastAPI
# ─────────────────────────────────────────────

def run_mcrs(
    target_user: int,
    weights: np.ndarray,
    train_df: pd.DataFrame,
    movies_df: pd.DataFrame,
    matrices: dict = None,
    n_neighbours: int = 30,
    n_recommendations: int = 10
) -> pd.DataFrame:
    """
    Entry point for the MCRS engine.
    Call this from FastAPI with the user's AGA-optimised weights.
    If matrices are not pre-built, builds them from train_df.
    """
    if matrices is None:
        matrices = build_user_item_matrices(train_df)

    recommendations = generate_recommendations(
        target_user=target_user,
        matrices=matrices,
        weights=weights,
        train_df=train_df,
        movies_df=movies_df,
        n_neighbours=n_neighbours,
        n_recommendations=n_recommendations
    )

    return recommendations