from modules.data_module  import build_dataset
from modules.evaluation   import run_evaluation

print("Loading dataset...")
data = build_dataset("data/ml-100k/ml-100k")

print("\nRunning full evaluation (50 users)...")
print("This will take a few minutes — the AGA runs once per user.\n")

summary = run_evaluation(
    data=data,
    n_users=1,
    n_neighbours=30,
    n_recommendations=10,
    like_threshold=4.0,
    verbose=True
)

print("\n" + "="*70)
print("EVALUATION RESULTS — AGA vs 3 Baselines")
print("="*70)
print(summary.to_string(index=False))
print("="*70)