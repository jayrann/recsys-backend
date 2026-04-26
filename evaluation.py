"""
evaluation.py
=============
Evaluation framework for the MCRS.

Metrics  : RMSE, MAE, Precision@N, Recall@N, F1@N, Coverage
Baselines:
    B1 — Single-criteria CF (overall rating only)
    B2 — Equal-weight MCRS (wc = 0.20 for all 5 criteria)
    B3 — Global regression MCRS (Ridge regression, one weight vector for all users)
"""

from __future__ import annotations


import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from typing import Optional

from modules.mcrs_engine import (
    build_user_item_matrices,
    find_neighbours,
    generate_recommendations,
    predict_rating,
    compute_utility,
    CRITERIA,
    NORM_COLS,
)
from modules.aga_module import run_aga


# ─────────────────────────────────────────────────────────────────────────────
# 1. PREDICTION ACCURACY METRICS
# ─────────────────────────────────────────────────────────────────────────────

def compute_rmse(predictions: list[float], actuals: list[float]) -> float:
    """Root Mean Square Error."""
    if not predictions:
        return float("nan")
    pred = np.array(predictions)
    act  = np.array(actuals)
    return float(np.sqrt(np.mean((pred - act) ** 2)))


def compute_mae(predictions: list[float], actuals: list[float]) -> float:
    """Mean Absolute Error."""
    if not predictions:
        return float("nan")
    pred = np.array(predictions)
    act  = np.array(actuals)
    return float(np.mean(np.abs(pred - act)))


# ─────────────────────────────────────────────────────────────────────────────
# 2. RANKING METRICS
# ─────────────────────────────────────────────────────────────────────────────

LIKE_THRESHOLD = 4.0   # a movie is "liked" if actual rating >= 4

def compute_precision_at_n(recommended_ids: list[int],
                            liked_ids: set[int]) -> float:
    """Precision@N = |liked in top-N| / N"""
    if not recommended_ids:
        return 0.0
    hits = sum(1 for mid in recommended_ids if mid in liked_ids)
    return hits / len(recommended_ids)


def compute_recall_at_n(recommended_ids: list[int],
                         liked_ids: set[int]) -> float:
    """Recall@N = |liked in top-N| / |all liked|"""
    if not liked_ids:
        return 0.0
    hits = sum(1 for mid in recommended_ids if mid in liked_ids)
    return hits / len(liked_ids)


def compute_f1(precision: float, recall: float) -> float:
    """F1 = 2 * P * R / (P + R)"""
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def compute_coverage(all_recommended: set[int],
                      total_movies: int) -> float:
    """Coverage = |unique recommended movies| / |total catalogue|"""
    if total_movies == 0:
        return 0.0
    return len(all_recommended) / total_movies


# ─────────────────────────────────────────────────────────────────────────────
# 3. BASELINE B1 — SINGLE-CRITERIA CF
# ─────────────────────────────────────────────────────────────────────────────

def build_single_criteria_matrix(train_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build user-item matrix using only the overall normalised rating.
    """
    mat = train_df.pivot_table(
        index="user_id",
        columns="movie_id",
        values="overall_norm",
        aggfunc="first"
    )
    return mat


def single_criteria_similarity(user_a: int,
                                user_b: int,
                                matrix: pd.DataFrame,
                                min_co_rated: int = 5) -> float:
    """Standard Pearson correlation on overall_norm ratings."""
    from scipy.stats import pearsonr
    if user_a not in matrix.index or user_b not in matrix.index:
        return 0.0

    row_a = matrix.loc[user_a]
    row_b = matrix.loc[user_b]
    co    = row_a.index[row_a.notna() & row_b.notna()]

    if len(co) < min_co_rated:
        return 0.0

    va, vb = row_a[co].values.astype(float), row_b[co].values.astype(float)

    if va.std() == 0 or vb.std() == 0:
        return 0.0

    corr, _ = pearsonr(va, vb)
    return float(np.clip(corr, -1.0, 1.0)) if not np.isnan(corr) else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# 4. BASELINE B3 — GLOBAL RIDGE REGRESSION WEIGHTS
# ─────────────────────────────────────────────────────────────────────────────

def learn_global_weights(train_df: pd.DataFrame) -> np.ndarray:
    """
    Fit Ridge regression to learn one weight vector for all users.

    X = [storyline_norm, acting_norm, visuals_norm, emotional_norm, enjoyment_norm]
    y = overall_norm

    Returns normalised weight vector (sum = 1).
    """
    X = train_df[NORM_COLS].values
    y = train_df["overall_norm"].values

    model = Ridge(alpha=1.0, fit_intercept=False)
    model.fit(X, y)

    weights = np.maximum(model.coef_, 0)  # non-negative constraint
    total   = weights.sum()
    if total == 0:
        weights = np.ones(5) / 5
    else:
        weights = weights / total

    print(f"[evaluation] Global Ridge weights: "
          f"{dict(zip(CRITERIA, weights.round(4)))}")
    return weights


# ─────────────────────────────────────────────────────────────────────────────
# 5. FULL EVALUATION FOR ONE USER
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_user(user_id: int,
                  train_df: pd.DataFrame,
                  test_df: pd.DataFrame,
                  matrices: dict,
                  weights: np.ndarray,
                  movies_df: pd.DataFrame,
                  top_n: int = 10) -> dict:
    """
    Evaluate a single model configuration for one user.

    Returns dict with RMSE, MAE, Precision@N, Recall@N, F1, and recommendation list.
    """
    user_test = test_df[test_df["user_id"] == user_id]
    if len(user_test) == 0:
        return None

    # ── Prediction accuracy ──────────────────────────────────────────────────
    neighbours = find_neighbours(user_id, matrices, weights, k=30)

    predictions = []
    actuals     = []

    for _, row in user_test.iterrows():
        movie_id = row["movie_id"]
        actual   = float(row["overall_norm"])
        pred     = predict_rating(user_id, movie_id, neighbours, matrices, weights)
        if pred is not None:
            predictions.append(pred)
            actuals.append(actual)

    rmse = compute_rmse(predictions, actuals)
    mae  = compute_mae(predictions, actuals)

    # ── Ranking metrics ───────────────────────────────────────────────────────
    recs     = generate_recommendations(user_id, neighbours, matrices,
                                        weights, movies_df, top_n)
    rec_ids  = [r["movie_id"] for r in recs]

    # "Liked" movies in test set (rating >= 4 on original scale)
    user_test_liked = user_test[user_test["overall_rating"] >= LIKE_THRESHOLD]
    liked_ids       = set(user_test_liked["movie_id"].tolist())

    precision = compute_precision_at_n(rec_ids, liked_ids)
    recall    = compute_recall_at_n(rec_ids, liked_ids)
    f1        = compute_f1(precision, recall)

    return {
        "user_id":          user_id,
        "n_predictions":    len(predictions),
        "rmse":             round(rmse, 4) if not np.isnan(rmse) else None,
        "mae":              round(mae, 4)  if not np.isnan(mae)  else None,
        "precision_at_n":   round(precision, 4),
        "recall_at_n":      round(recall, 4),
        "f1":               round(f1, 4),
        "rec_ids":          rec_ids,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 6. RUN EVALUATION — ALL MODELS
# ─────────────────────────────────────────────────────────────────────────────

def run_evaluation(train_df: pd.DataFrame,
                   test_df: pd.DataFrame,
                   movies_df: pd.DataFrame,
                   target_user: Optional[int] = None,
                   n_users: int = 20,
                   top_n: int = 10) -> dict:
    """
    Run comparative evaluation of 4 models:
        AGA-MCRS, B1 (single-criteria), B2 (equal weights), B3 (ridge)

    If target_user is given, evaluate only that user.
    Otherwise evaluate n_users randomly sampled users.

    Returns dict of model_name → {avg_rmse, avg_mae, avg_precision,
                                   avg_recall, avg_f1, coverage}
    """
    print("\n" + "="*60)
    print("[evaluation] Starting comparative evaluation")
    print("="*60)

    # Select users to evaluate
    if target_user is not None:
        user_ids = [target_user]
    else:
        all_test_users = test_df["user_id"].unique().tolist()
        n_users        = min(n_users, len(all_test_users))
        rng            = np.random.default_rng(42)
        user_ids       = rng.choice(all_test_users, size=n_users,
                                     replace=False).tolist()

    print(f"[evaluation] Evaluating {len(user_ids)} users")

    # Build matrices for MCRS (used by B2, B3, AGA)
    matrices = build_user_item_matrices(train_df)
    total_movies = movies_df["movie_id"].nunique()

    # B2 equal weights
    equal_weights = np.ones(5) / 5

    # B3 global Ridge weights
    global_weights = learn_global_weights(train_df)

    # Results containers
    results = {
        "AGA":    [],
        "B1_single_criteria": [],
        "B2_equal_weights":   [],
        "B3_global_ridge":    [],
    }

    all_recs = {k: set() for k in results}

    for user_id in user_ids:
        user_train = train_df[train_df["user_id"] == user_id]
        if len(user_train) < 5:
            continue

        print(f"  → User {user_id} ({len(user_train)} training ratings)")

        # ── AGA weights ───────────────────────────────────────────────────────
        aga_result  = run_aga(user_id, user_train)
        aga_weights = aga_result["best_weights"]

        # ── Evaluate AGA ──────────────────────────────────────────────────────
        r_aga = evaluate_user(user_id, train_df, test_df,
                               matrices, aga_weights, movies_df, top_n)
        if r_aga:
            results["AGA"].append(r_aga)
            all_recs["AGA"].update(r_aga["rec_ids"])

        # ── Evaluate B2 (equal weights) ───────────────────────────────────────
        r_b2 = evaluate_user(user_id, train_df, test_df,
                              matrices, equal_weights, movies_df, top_n)
        if r_b2:
            results["B2_equal_weights"].append(r_b2)
            all_recs["B2_equal_weights"].update(r_b2["rec_ids"])

        # ── Evaluate B3 (global ridge) ────────────────────────────────────────
        r_b3 = evaluate_user(user_id, train_df, test_df,
                              matrices, global_weights, movies_df, top_n)
        if r_b3:
            results["B3_global_ridge"].append(r_b3)
            all_recs["B3_global_ridge"].update(r_b3["rec_ids"])

        # ── Evaluate B1 (single-criteria CF) ──────────────────────────────────
        # Build single-criteria matrices with equal weights on overall_norm only
        sc_weights = np.zeros(5)  # will not use these for prediction
        # For B1 we build a special single-criteria matrix
        sc_matrix  = {"overall": build_single_criteria_matrix(train_df)}

        # Build a surrogate 5-criterion matrix with overall repeated
        sc_matrices = {c: train_df.pivot_table(
            index="user_id", columns="movie_id",
            values="overall_norm", aggfunc="first"
        ) for c in CRITERIA}

        r_b1 = evaluate_user(user_id, train_df, test_df,
                              sc_matrices, equal_weights, movies_df, top_n)
        if r_b1:
            results["B1_single_criteria"].append(r_b1)
            all_recs["B1_single_criteria"].update(r_b1["rec_ids"])

    # ── Aggregate results ─────────────────────────────────────────────────────
    summary = {}
    for model_name, user_results in results.items():
        if not user_results:
            summary[model_name] = {"no_results": True}
            continue

        rmse_vals = [r["rmse"] for r in user_results if r["rmse"] is not None]
        mae_vals  = [r["mae"]  for r in user_results if r["mae"]  is not None]
        prec_vals = [r["precision_at_n"] for r in user_results]
        rec_vals  = [r["recall_at_n"]    for r in user_results]
        f1_vals   = [r["f1"]             for r in user_results]

        coverage  = compute_coverage(all_recs[model_name], total_movies)

        summary[model_name] = {
            "n_users_evaluated": len(user_results),
            "avg_rmse":          round(float(np.mean(rmse_vals)), 4) if rmse_vals else None,
            "avg_mae":           round(float(np.mean(mae_vals)),  4) if mae_vals  else None,
            "avg_precision":     round(float(np.mean(prec_vals)), 4),
            "avg_recall":        round(float(np.mean(rec_vals)),  4),
            "avg_f1":            round(float(np.mean(f1_vals)),   4),
            "coverage":          round(coverage, 4),
        }

    # ── Print comparison table ────────────────────────────────────────────────
    print("\n" + "="*70)
    print(f"{'Model':<25} {'RMSE':>6} {'MAE':>6} {'P@10':>6} "
          f"{'R@10':>6} {'F1':>6} {'Cov':>6}")
    print("-"*70)
    for model_name, s in summary.items():
        if "no_results" in s:
            continue
        print(f"{model_name:<25} "
              f"{str(s['avg_rmse']):>6} "
              f"{str(s['avg_mae']):>6} "
              f"{str(s['avg_precision']):>6} "
              f"{str(s['avg_recall']):>6} "
              f"{str(s['avg_f1']):>6} "
              f"{str(s['coverage']):>6}")
    print("="*70 + "\n")

    return summary
