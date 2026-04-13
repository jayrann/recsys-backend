import numpy as np
import pandas as pd
from modules.data_module import build_dataset
from modules.mcrs_engine import build_user_item_matrices, run_mcrs
from modules.aga_module  import run_aga

# ─────────────────────────────────────────────
# SECTION 3.7 — Evaluation Metrics
# RMSE, MAE, Precision@N, Recall@N, F1, Coverage
# ─────────────────────────────────────────────

CRITERIA = [
    "storyline_norm",
    "acting_norm",
    "visuals_norm",
    "emotional_impact_norm",
    "enjoyment_norm"
]


# ─────────────────────────────────────────────
# Metric 1 — RMSE
# Measures average magnitude of prediction errors
# Penalises large errors more heavily due to squaring
# ─────────────────────────────────────────────

def compute_rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    """
    RMSE = sqrt( (1/N) * SUM[ (predicted - actual)^2 ] )
    Lower is better.
    """
    if len(actual) == 0:
        return None
    return float(np.sqrt(np.mean((predicted - actual) ** 2)))


# ─────────────────────────────────────────────
# Metric 2 — MAE
# Average absolute prediction error
# More interpretable than RMSE — direct error in rating points
# ─────────────────────────────────────────────

def compute_mae(actual: np.ndarray, predicted: np.ndarray) -> float:
    """
    MAE = (1/N) * SUM[ |predicted - actual| ]
    Lower is better.
    """
    if len(actual) == 0:
        return None
    return float(np.mean(np.abs(predicted - actual)))


# ─────────────────────────────────────────────
# Metric 3 & 4 — Precision@N and Recall@N
# Precision = fraction of top-N recs the user actually likes
# Recall    = fraction of all liked movies that appear in top-N
# A movie is "liked" if actual rating >= 4 (Chapter 3.7)
# ─────────────────────────────────────────────

def compute_precision_recall(
    recommended_ids: list,
    actual_liked_ids: set,
) -> tuple:
    """
    Precision@N = |recommended ∩ liked| / |recommended|
    Recall@N    = |recommended ∩ liked| / |liked|
    Returns (precision, recall) tuple.
    """
    if not recommended_ids:
        return 0.0, 0.0

    recommended_set = set(recommended_ids)
    hits = len(recommended_set & actual_liked_ids)

    precision = hits / len(recommended_ids)
    recall    = hits / len(actual_liked_ids) if actual_liked_ids else 0.0

    return float(precision), float(recall)


# ─────────────────────────────────────────────
# Metric 5 — F1 Score
# Harmonic mean of Precision and Recall
# Penalises models strong on one but weak on the other
# ─────────────────────────────────────────────

def compute_f1(precision: float, recall: float) -> float:
    """
    F1 = 2 * (Precision * Recall) / (Precision + Recall)
    Returns 0.0 if both are zero.
    """
    if precision + recall == 0:
        return 0.0
    return float(2 * (precision * recall) / (precision + recall))


# ─────────────────────────────────────────────
# Metric 6 — Coverage
# What percentage of all movies the system ever recommends
# Low coverage = filter bubble (only popular movies recommended)
# ─────────────────────────────────────────────

def compute_coverage(
    all_recommended_ids: set,
    total_movies: int
) -> float:
    """
    Coverage = |unique recommended movies| / |total movies|
    Higher is better.
    """
    if total_movies == 0:
        return 0.0
    return float(len(all_recommended_ids) / total_movies)


# ─────────────────────────────────────────────
# Rating Prediction Evaluation
# Generate predicted vs actual pairs for RMSE/MAE
# ─────────────────────────────────────────────

def evaluate_predictions(
    user_id: int,
    test_df: pd.DataFrame,
    train_df: pd.DataFrame,
    matrices: dict,
    weights: np.ndarray,
    n_neighbours: int = 30
) -> dict:
    """
    For a given user, predict ratings for their test movies
    and compute RMSE and MAE against actual ratings.
    Returns dict with rmse, mae, n_predictions.
    """
    from modules.mcrs_engine import find_neighbours, predict_rating

    user_test = test_df[test_df["user_id"] == user_id]

    if len(user_test) == 0:
        return {"rmse": None, "mae": None, "n_predictions": 0}

    neighbours = find_neighbours(
        user_id, matrices, weights, n_neighbours
    )

    if not neighbours:
        return {"rmse": None, "mae": None, "n_predictions": 0}

    actuals    = []
    predicteds = []

    for _, row in user_test.iterrows():
        movie_id = row["movie_id"]
        actual = (row["rating"] - 1) / 4.0 

        predicted = predict_rating(
            user_id, movie_id, neighbours,
            matrices, weights, train_df
        )

        if predicted is not None:
            actuals.append(actual)
            predicteds.append(predicted)

    if not actuals:
        return {"rmse": None, "mae": None, "n_predictions": 0}

    actuals    = np.array(actuals)
    predicteds = np.array(predicteds)

    return {
        "rmse"          : compute_rmse(actuals, predicteds),
        "mae"           : compute_mae(actuals, predicteds),
        "n_predictions" : len(actuals)
    }


# ─────────────────────────────────────────────
# Baseline Models (Chapter 3.7)
# Three baselines to compare AGA against
# ─────────────────────────────────────────────

def get_baseline_weights(baseline: str, train_df: pd.DataFrame,
                         user_id: int) -> np.ndarray:
    """
    Return weights for each of the 3 baseline models:

    Baseline 1 — Single-criteria CF:
        Use only the overall rating (equal weights, single criterion proxy)

    Baseline 2 — Multi-criteria equal weights:
        All 5 criteria weighted equally at 0.2
        Tests whether AGA contributes beyond just multi-criteria data

    Baseline 3 — Global linear regression weights:
        One-size-fits-all weights from linear regression across all users
        Tests whether personalised weights outperform global weights
    """
    if baseline == "single_criteria":
        # Proxy: heavily weight one criterion to simulate single-rating CF
        return np.array([0.5, 0.125, 0.125, 0.125, 0.125])

    elif baseline == "equal_weights":
        return np.ones(5) / 5.0

    elif baseline == "global_regression":
        # Fit linear regression on all training data to find global weights
        from sklearn.linear_model import Ridge

        X = train_df[CRITERIA].values
        y = (train_df["rating"].values - 1) / 4.0  # normalise to [0,1]

        # Ridge regression to avoid overfitting
        reg = Ridge(alpha=1.0, fit_intercept=False, positive=True)
        reg.fit(X, y)

        weights = reg.coef_
        weights = np.abs(weights)

        # Normalise to sum to 1
        total = weights.sum()
        if total > 0:
            weights = weights / total
        else:
            weights = np.ones(5) / 5.0

        return weights

    else:
        raise ValueError(f"Unknown baseline: {baseline}")


# ─────────────────────────────────────────────
# Full Evaluation Pipeline
# Runs all metrics across a sample of users
# Compares AGA vs 3 baselines
# ─────────────────────────────────────────────

def run_evaluation(
    data: dict,
    n_users: int = 50,
    n_neighbours: int = 30,
    n_recommendations: int = 10,
    like_threshold: float = 4.0,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Full evaluation pipeline from Chapter 3.7.
    Runs across n_users sampled from the dataset.
    Compares AGA weights against 3 baseline models.
    Returns a summary DataFrame with all metrics.
    """
    train_df  = data["train_df"]
    test_df   = data["test_df"]
    movies_df = data["movies_df"]
    total_movies = movies_df["movie_id"].nunique()

    # Build matrices once
    if verbose:
        print("[EVAL] Building user-item matrices...")
    matrices = build_user_item_matrices(train_df)

    # Sample users who have enough test ratings
    user_test_counts = test_df["user_id"].value_counts()
    eligible_users   = user_test_counts[user_test_counts >= 3].index.tolist()
    sample_users     = eligible_users[:n_users]

    if verbose:
        print(f"[EVAL] Evaluating {len(sample_users)} users across "
              f"4 models (AGA + 3 baselines)...")

    # Tracking metrics per model
    models = ["aga", "equal_weights", "single_criteria", "global_regression"]
    results = {m: {
        "rmse_list"     : [],
        "mae_list"      : [],
        "precision_list": [],
        "recall_list"   : [],
        "f1_list"       : [],
        "recommended_ids": set()
    } for m in models}

    global_reg_weights = get_baseline_weights(
        "global_regression", train_df, None
    )

    for i, user_id in enumerate(sample_users):
        if verbose and (i + 1) % 10 == 0:
            print(f"[EVAL] Processing user {i+1}/{len(sample_users)}...")

        user_train = train_df[train_df["user_id"] == user_id]
        user_test  = test_df[test_df["user_id"] == user_id]

        # Movies the user actually liked in test set
        liked_ids = set(
            user_test[user_test["rating"] >= like_threshold]["movie_id"]
            .tolist()
            )

        # Get weights for each model
        weight_map = {
            "equal_weights"     : np.ones(5) / 5.0,
            "single_criteria"   : get_baseline_weights(
                                    "single_criteria", train_df, user_id),
            "global_regression" : global_reg_weights,
        }

        # Run AGA for this user
        aga_result = run_aga(
            user_id=user_id,
            user_ratings=user_train,
            verbose=False
        )
        weight_map["aga"] = aga_result["best_weights"]

        # Evaluate each model
        for model in models:
            weights = weight_map[model]

            # RMSE and MAE
            pred_metrics = evaluate_predictions(
                user_id, test_df, train_df,
                matrices, weights, n_neighbours
            )

            if pred_metrics["rmse"] is not None:
                results[model]["rmse_list"].append(pred_metrics["rmse"])
                results[model]["mae_list"].append(pred_metrics["mae"])

            # Precision, Recall, F1
            recs = run_mcrs(
                target_user=user_id,
                weights=weights,
                train_df=train_df,
                movies_df=movies_df,
                matrices=matrices,
                n_recommendations=n_recommendations
            )

            if not recs.empty:
                rec_ids = recs["movie_id"].tolist()
                results[model]["recommended_ids"].update(rec_ids)

                if liked_ids:
                    precision, recall = compute_precision_recall(
                        rec_ids, liked_ids
                    )
                    f1 = compute_f1(precision, recall)
                    results[model]["precision_list"].append(precision)
                    results[model]["recall_list"].append(recall)
                    results[model]["f1_list"].append(f1)

    # ── Compile summary table ──
    summary_rows = []
    model_labels = {
        "aga"              : "AGA (proposed)",
        "equal_weights"    : "Baseline 1 — Equal weights",
        "single_criteria"  : "Baseline 2 — Single criteria",
        "global_regression": "Baseline 3 — Global regression"
    }

    for model in models:
        r = results[model]

        rmse     = np.mean(r["rmse_list"])     if r["rmse_list"]      else None
        mae      = np.mean(r["mae_list"])      if r["mae_list"]       else None
        prec     = np.mean(r["precision_list"]) if r["precision_list"] else None
        recall   = np.mean(r["recall_list"])   if r["recall_list"]    else None
        f1       = np.mean(r["f1_list"])       if r["f1_list"]        else None
        coverage = compute_coverage(
            r["recommended_ids"], total_movies
        )

        summary_rows.append({
            "Model"       : model_labels[model],
            "RMSE"        : round(rmse, 4)     if rmse     else "N/A",
            "MAE"         : round(mae, 4)      if mae      else "N/A",
            "Precision@10": round(prec, 4)     if prec     else "N/A",
            "Recall@10"   : round(recall, 4)   if recall   else "N/A",
            "F1"          : round(f1, 4)       if f1       else "N/A",
            "Coverage"    : round(coverage, 4)
        })

    summary_df = pd.DataFrame(summary_rows)
    return summary_df