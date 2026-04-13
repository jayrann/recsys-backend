import numpy as np
from modules.data_module import build_dataset
from modules.mcrs_engine import build_user_item_matrices, run_mcrs

# Load and preprocess data
print("Loading dataset...")
data = build_dataset("data/ml-100k/ml-100k")

train_df  = data["train_df"]
movies_df = data["movies_df"]

# Pre-build matrices once (expensive, do this once at startup)
print("\nBuilding user-item matrices...")
matrices = build_user_item_matrices(train_df)

# Test with user 1 using equal weights (0.2 each)
# Later the AGA will replace these with optimised weights
target_user = 1
equal_weights = np.array([0.2, 0.2, 0.2, 0.2, 0.2])

print(f"\nRunning MCRS for user {target_user} with equal weights...")
recommendations = run_mcrs(
    target_user=target_user,
    weights=equal_weights,
    train_df=train_df,
    movies_df=movies_df,
    matrices=matrices
)

print("\n── Top 10 Recommendations for User 1 ──")
print(recommendations[["title", "predicted_score"]].to_string())