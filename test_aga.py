import numpy as np
from modules.data_module import build_dataset
from modules.mcrs_engine import build_user_item_matrices, run_mcrs
from modules.aga_module  import run_aga

# ── Load data ──
print("Loading dataset...")
data      = build_dataset("data/ml-100k/ml-100k")
train_df  = data["train_df"]
movies_df = data["movies_df"]

# ── Run AGA for user 1 ──
target_user  = 1
user_ratings = train_df[train_df["user_id"] == target_user]

result = run_aga(
    user_id      = target_user,
    user_ratings = user_ratings,
    verbose      = True
)

aga_weights = result["best_weights"]

# ── Build matrices ──
print("\nBuilding user-item matrices...")
matrices = build_user_item_matrices(train_df)

# ── Compare equal weights vs AGA weights ──
print("\n── Recommendations with EQUAL weights (0.2 each) ──")
equal_weights = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
recs_equal = run_mcrs(
    target_user=target_user,
    weights=equal_weights,
    train_df=train_df,
    movies_df=movies_df,
    matrices=matrices
)
print(recs_equal[["title", "predicted_score"]].to_string())

print("\n── Recommendations with AGA-OPTIMISED weights ──")
recs_aga = run_mcrs(
    target_user=target_user,
    weights=aga_weights,
    train_df=train_df,
    movies_df=movies_df,
    matrices=matrices
)
print(recs_aga[["title", "predicted_score"]].to_string())