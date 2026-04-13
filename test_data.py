from modules.data_module import build_dataset

data = build_dataset("data/ml-100k/ml-100k")

print("\nSample of processed ratings:")
print(data["full_df"][["user_id", "movie_id", "rating",
                         "storyline", "acting", "visuals",
                         "emotional_impact", "enjoyment",
                         "storyline_norm", "split"]].head(10))

print("\nSample of movies:")
print(data["movies_df"][["movie_id", "title"]].head(5))