"""
train.py
========
Explicit training pipeline for the MCRS.

This script demonstrates the "training" phase of the system:
  1. Load and preprocess the MovieLens dataset (train/test split)
  2. For each user in the training set, run the AGA to learn their
     optimal criterion weight vector W* from their training ratings
  3. Save all weight profiles to a JSON file (weight_profiles.json)
     — this simulates what would be stored in the WeightProfiles MySQL table

In a live deployment, this script would be run:
  - Once at system launch (for all existing users)
  - Periodically for users who have submitted new ratings
  - On demand via the POST /admin/retrain endpoint

Run:
    python train.py --users 50
    python train.py --users all
    python train.py --user 1        (single user)
"""

from __future__ import annotations


import os
import sys
import json
import time
import argparse
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))

from modules.data_module import build_dataset
from modules.aga_module  import run_aga, format_weights, AGA_CONFIG

DATA_DIR        = "data/ml-100k/ml-100k"
PROFILES_FILE   = "weight_profiles.json"


# ─────────────────────────────────────────────────────────────────────────────
# TRAINING PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def train(n_users=None, target_user=None):
    """
    Run AGA weight optimisation for users in the training set.

    Parameters
    ----------
    n_users     : int or None — how many users to train (None = all)
    target_user : int or None — train only this specific user
    """

    print("\n" + "="*60)
    print("MCRS TRAINING PIPELINE")
    print("="*60)

    # ── Step 1: Load and preprocess dataset ───────────────────────────────────
    print("\n[Step 1] Loading and preprocessing dataset...")
    dataset  = build_dataset(DATA_DIR, version="100k")
    train_df = dataset["train_df"]
    test_df  = dataset["test_df"]

    print(f"\n  Training set : {len(train_df):,} ratings across "
          f"{train_df['user_id'].nunique()} users")
    print(f"  Test set     : {len(test_df):,} ratings across "
          f"{test_df['user_id'].nunique()} users")
    print(f"\n  NOTE: The test set is NEVER seen during AGA optimisation.")
    print(f"  The AGA only uses train_df to evaluate fitness (MAE).")

    # ── Step 2: Select users to train ─────────────────────────────────────────
    all_users = sorted(train_df["user_id"].unique().tolist())

    if target_user is not None:
        users_to_train = [target_user]
    elif n_users is None or n_users == "all":
        users_to_train = all_users
    else:
        users_to_train = all_users[:int(n_users)]

    print(f"\n[Step 2] Training AGA weight profiles for "
          f"{len(users_to_train)} user(s)...")
    print(f"  AGA config: population={AGA_CONFIG['population_size']}, "
          f"max_gen={AGA_CONFIG['max_generations']}, "
          f"Pc={AGA_CONFIG['crossover_rate']}, "
          f"Pm_base={AGA_CONFIG['mutation_base']}\n")

    # ── Step 3: Run AGA per user ───────────────────────────────────────────────
    profiles    = {}
    start_total = time.time()

    for i, user_id in enumerate(users_to_train):
        user_train = train_df[train_df["user_id"] == user_id]

        if len(user_train) < 5:
            print(f"  [User {user_id:4}] SKIP — only {len(user_train)} ratings "
                  f"(minimum 5 required)")
            # Fallback: equal weights
            profiles[int(user_id)] = {
                "user_id":       int(user_id),
                "w1_storyline":  0.2,
                "w2_acting":     0.2,
                "w3_visuals":    0.2,
                "w4_emotional":  0.2,
                "w5_enjoyment":  0.2,
                "best_mae":      None,
                "generations":   0,
                "converged":     False,
                "source":        "equal_weights_fallback"
            }
            continue

        t0     = time.time()
        result = run_aga(user_id, user_train)
        t1     = time.time()
        fmt    = format_weights(result)
        fmt["source"] = "aga_optimised"

        profiles[int(user_id)] = fmt

        # Progress display
        bar = "█" * int((i + 1) / len(users_to_train) * 20)
        print(f"  [{i+1:3}/{len(users_to_train)}] User {user_id:4} | "
              f"MAE={fmt['best_mae']:.4f} | "
              f"gen={fmt['generations']:3} | "
              f"converged={fmt['converged']} | "
              f"{t1-t0:.1f}s | "
              f"w=[{fmt['w1_storyline']:.3f}, {fmt['w2_acting']:.3f}, "
              f"{fmt['w3_visuals']:.3f}, {fmt['w4_emotional']:.3f}, "
              f"{fmt['w5_enjoyment']:.3f}]")

    total_time = time.time() - start_total

    # ── Step 4: Save weight profiles ──────────────────────────────────────────
    print(f"\n[Step 3] Saving {len(profiles)} weight profiles → {PROFILES_FILE}")

    with open(PROFILES_FILE, "w") as f:
        json.dump(profiles, f, indent=2)

    print(f"  Saved successfully.")

    # ── Step 5: Summary statistics ────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("TRAINING SUMMARY")
    print(f"{'='*60}")

    aga_profiles  = [p for p in profiles.values() if p["source"] == "aga_optimised"]
    fallback      = [p for p in profiles.values() if "fallback" in p["source"]]

    if aga_profiles:
        maes = [p["best_mae"] for p in aga_profiles]
        gens = [p["generations"] for p in aga_profiles]
        conv = sum(1 for p in aga_profiles if p["converged"])

        print(f"  Users trained (AGA)    : {len(aga_profiles)}")
        print(f"  Users (equal fallback) : {len(fallback)}")
        print(f"  Avg best MAE           : {np.mean(maes):.4f}")
        print(f"  Min MAE                : {np.min(maes):.4f}")
        print(f"  Max MAE                : {np.max(maes):.4f}")
        print(f"  Avg generations        : {np.mean(gens):.1f}")
        print(f"  Early-stop convergence : {conv}/{len(aga_profiles)} users")
        print(f"  Total training time    : {total_time:.1f}s")
        print(f"  Avg time per user      : {total_time/len(aga_profiles):.2f}s")

        # Show most differentiated weight profiles
        print(f"\n  Top 5 most differentiated weight profiles:")
        print(f"  {'User':>6}  {'Story':>6}  {'Act':>6}  {'Vis':>6}  "
              f"{'Emot':>6}  {'Enj':>6}  {'MAE':>7}")
        print(f"  {'-'*52}")
        # Sort by variance of weights (most differentiated first)
        def weight_variance(p):
            w = [p["w1_storyline"], p["w2_acting"], p["w3_visuals"],
                 p["w4_emotional"], p["w5_enjoyment"]]
            return np.var(w)

        top5 = sorted(aga_profiles, key=weight_variance, reverse=True)[:5]
        for p in top5:
            print(f"  {p['user_id']:>6}  "
                  f"{p['w1_storyline']:>6.3f}  "
                  f"{p['w2_acting']:>6.3f}  "
                  f"{p['w3_visuals']:>6.3f}  "
                  f"{p['w4_emotional']:>6.3f}  "
                  f"{p['w5_enjoyment']:>6.3f}  "
                  f"{p['best_mae']:>7.4f}")

    print(f"\n  Weight profiles saved to: {os.path.abspath(PROFILES_FILE)}")
    print(f"{'='*60}\n")
    return profiles


# ─────────────────────────────────────────────────────────────────────────────
# LOAD SAVED PROFILES (used by main.py / FastAPI)
# ─────────────────────────────────────────────────────────────────────────────

def load_weight_profiles(filepath=PROFILES_FILE):
    """
    Load previously trained weight profiles from JSON.
    Returns dict: {user_id_int: {w1..w5, mae, gen, ...}}
    """
    if not os.path.exists(filepath):
        return {}
    with open(filepath, "r") as f:
        raw = json.load(f)
    # Keys are strings in JSON; convert back to int
    return {int(k): v for k, v in raw.items()}


def get_weights_for_user(user_id, profiles):
    """
    Retrieve np.ndarray of weights for a user from saved profiles.
    Falls back to equal weights if user not found.
    """
    profile = profiles.get(int(user_id))
    if profile is None:
        return np.ones(5) / 5
    return np.array([
        profile["w1_storyline"],
        profile["w2_acting"],
        profile["w3_visuals"],
        profile["w4_emotional"],
        profile["w5_enjoyment"]
    ])


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train AGA weight profiles for MCRS users"
    )
    parser.add_argument(
        "--users",
        default="10",
        help="Number of users to train (integer), 'all', default=10"
    )
    parser.add_argument(
        "--user",
        type=int,
        default=None,
        help="Train a single specific user by ID"
    )
    args = parser.parse_args()

    if args.user is not None:
        train(target_user=args.user)
    elif args.users == "all":
        train(n_users=None)
    else:
        train(n_users=int(args.users))
