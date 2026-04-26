"""
main.py
=======
FastAPI backend for the Multi-Criteria Recommender System.

Architecture:
  - JWT authentication (auth.py)
  - MySQL database layer (database.py)
  - Dataset loaded once at startup into memory for matrix operations
  - AGA weight profiles cached in MySQL WeightProfiles table

Run:
    uvicorn main:app --reload --port 8000

Interactive docs:
    http://localhost:8000/docs
"""

from __future__ import annotations


import os
import json
import numpy as np
import pandas as pd
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel, Field
from typing import Optional

from config import load_dotenv, settings
from auth import (
    hash_password, verify_password,
    create_access_token,
    get_current_user, require_admin
)
from database import (
    test_connection, init_db,
    UserDB, MovieDB, RatingDB,
    WeightProfileDB, RecommendationLogDB
)
from modules.data_module  import build_dataset
from modules.mcrs_engine  import build_user_item_matrices, run_mcrs
from modules.aga_module   import run_aga
from train import load_weight_profiles

load_dotenv()


# ─────────────────────────────────────────────────────────────────────────────
# APPLICATION STATE
# ─────────────────────────────────────────────────────────────────────────────

class AppState:
    dataset:      dict = {}
    matrices:     dict = {}
    use_db:       bool = False
    weight_cache: dict = {}


state = AppState()


# ─────────────────────────────────────────────────────────────────────────────
# STARTUP / SHUTDOWN
# ─────────────────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("\n[main] Starting MCRS server...")

    state.use_db = test_connection()
    if state.use_db:
        print("[main] MySQL connected.")
        init_db()
    else:
        print("[main] MySQL unavailable — running in demo mode.")

    data_dir = settings.movielens_dir
    if os.path.exists(data_dir):
        print(f"[main] Loading dataset from {data_dir}...")
        state.dataset = build_dataset(data_dir)
        state.matrices = build_user_item_matrices(state.dataset["train_df"])
        print("[main] Dataset ready.")
    else:
        print(f"[main] Dataset not found at {data_dir}")

    profiles = load_weight_profiles(settings.weight_profiles)
    for uid, p in profiles.items():
        state.weight_cache[int(uid)] = np.array([
            p["w1_storyline"], p["w2_acting"], p["w3_visuals"],
            p["w4_emotional"], p["w5_enjoyment"]
        ])
    print(f"[main] Loaded {len(state.weight_cache)} weight profiles.")

    yield

    print("[main] Shutting down.")


app = FastAPI(
    title="MCRS — Multi-Criteria Recommender System",
    description="Improving Multi-Criteria Recommender Systems Using Adaptive Genetic Algorithms",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────────────────────────────────────
# PYDANTIC SCHEMAS
# ─────────────────────────────────────────────────────────────────────────────

class RegisterRequest(BaseModel):
    username: str  = Field(..., min_length=3, max_length=50)
    email:    str  = Field(..., min_length=5, max_length=100)
    password: str  = Field(..., min_length=6)
    role:     str  = Field(default="user")

class RatingRequest(BaseModel):
    movie_id:         int
    storyline:        float = Field(..., ge=1, le=5)
    acting:           float = Field(..., ge=1, le=5)
    visuals:          float = Field(..., ge=1, le=5)
    emotional_impact: float = Field(..., ge=1, le=5)
    enjoyment:        float = Field(..., ge=1, le=5)
    
class MovieUpsertRequest(BaseModel):
    movie_id:     int
    title:        str
    release_year: Optional[int] = None
    genres:       Optional[str] = None
    synopsis:     Optional[str] = None
    poster_url:   Optional[str] = None


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _require_dataset():
    if not state.dataset:
        raise HTTPException(503, "Dataset not loaded.")


def _get_weights(user_id: int) -> np.ndarray:
    """
    Return AGA-optimised weights for a user.
    Priority: MySQL → in-memory cache → compute AGA → equal fallback.
    """
    equal_weights = np.ones(5) / 5

    # 1. Try MySQL cached profile
    if state.use_db:
        profile = WeightProfileDB.get(user_id)
        if profile:
            w = np.array([
                float(profile["w1_storyline"]),
                float(profile["w2_acting"]),
                float(profile["w3_visuals"]),
                float(profile["w4_emotional"]),
                float(profile["w5_enjoyment"]),
            ])
            state.weight_cache[user_id] = w
            return w

    # 2. Try in-memory cache
    if user_id in state.weight_cache:
        return state.weight_cache[user_id]

    # 3. Run AGA on MySQL ratings (real user ratings)
    if state.use_db:
        import pandas as pd
        user_ratings = RatingDB.get_user_training_ratings(user_id)
        if len(user_ratings) >= 5:
            user_train = pd.DataFrame(user_ratings)
            for col in user_train.columns:
                try:
                    user_train[col] = user_train[col].astype(float)
                except (ValueError, TypeError):
                    pass
            if "overall_rating" in user_train.columns:
                user_train = user_train.rename(columns={"overall_rating": "rating"})
            result = run_aga(user_id, user_train)
            w = result["best_weights"]
            state.weight_cache[user_id] = w
            WeightProfileDB.upsert(
                user_id=user_id,
                w1=float(w[0]), w2=float(w[1]),
                w3=float(w[2]), w4=float(w[3]),
                w5=float(w[4]),
                best_mae=result.get("best_mae", 0.0),
                generations=result.get("generations", 0),
                converged=result.get("converged", False)
            )
            return w
        else:
            return equal_weights

    # 4. Fall back to MovieLens dataset (only when no MySQL)
    if state.dataset:
        train_df = state.dataset["train_df"]
        user_train = train_df[train_df["user_id"] == user_id]
        if len(user_train) >= 5:
            result = run_aga(user_id, user_train)
            w = result["best_weights"]
            state.weight_cache[user_id] = w
            return w

    return equal_weights
# ─────────────────────────────────────────────────────────────────────────────
# ROOT
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {
        "system":         "Multi-Criteria Recommender System",
        "version":        "1.0.0",
        "status":         "running",
        "database":       "connected" if state.use_db else "offline (demo mode)",
        "dataset_loaded": bool(state.dataset),
        "docs":           "/docs"
    }


# ─────────────────────────────────────────────────────────────────────────────
# AUTH
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/auth/register", status_code=201)
def register(req: RegisterRequest):
    if req.role not in ("user", "admin"):
        raise HTTPException(400, "role must be 'user' or 'admin'")
    hashed = hash_password(req.password)
    if state.use_db:
        try:
            import pymysql
            user_id = UserDB.create(req.username, req.email, hashed, req.role)
        except pymysql.IntegrityError:
            raise HTTPException(400, "Username or email already registered")
    else:
        user_id = abs(hash(req.username)) % 100000
    return {"message": "Account created", "user_id": user_id,
            "username": req.username}


@app.post("/auth/login")
def login(form: OAuth2PasswordRequestForm = Depends()):
    if state.use_db:
        user = UserDB.get_by_username(form.username)
        if not user or not verify_password(form.password, user["password_hash"]):
            raise HTTPException(
                status.HTTP_401_UNAUTHORIZED,
                "Incorrect username or password",
                headers={"WWW-Authenticate": "Bearer"}
            )
        UserDB.update_last_login(user["user_id"])
        user_id, role = user["user_id"], user["role"]
    else:
        user_id, role = 1, "admin"   # demo mode

    token = create_access_token({"sub": str(user_id), "role": role})
    return {"access_token": token, "token_type": "bearer",
            "user_id": user_id, "role": role}


# ─────────────────────────────────────────────────────────────────────────────
# RECOMMENDATIONS
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/recommendations/{user_id}")
def get_recommendations(
    user_id: int,
    top_n:   int  = 10,
    current: dict = Depends(get_current_user)
):
    _require_dataset()
    weights = _get_weights(user_id)

    # Inject new user ratings into matrices if user not in MovieLens dataset
    import pandas as pd
    if user_id not in state.matrices["storyline_norm"].index:
        if state.use_db:
            user_ratings = RatingDB.get_user_training_ratings(user_id)
            if user_ratings:
                user_df = pd.DataFrame(user_ratings)
                for col in user_df.columns:
                    try:
                        user_df[col] = user_df[col].astype(float)
                    except:
                        pass
                for _, row in user_df.iterrows():
                    mid = int(row["movie_id"])
                    state.matrices["storyline_norm"].loc[user_id, mid] = float(row["storyline_norm"])
                    state.matrices["acting_norm"].loc[user_id, mid] = float(row["acting_norm"])
                    state.matrices["visuals_norm"].loc[user_id, mid] = float(row["visuals_norm"])
                    state.matrices["emotional_impact_norm"].loc[user_id, mid] = float(row["emotional_impact_norm"])
                    state.matrices["enjoyment_norm"].loc[user_id, mid] = float(row["enjoyment_norm"])
                print(f"[main] Injected {len(user_df)} ratings for new user {user_id} into matrices")

    result = run_mcrs(
        target_user=user_id,
        weights=weights,
        train_df=state.dataset["train_df"],
        movies_df=state.dataset["movies_df"],
        matrices=state.matrices,
        n_neighbours=30,
        n_recommendations=top_n
    )
    
    print(f"[DEBUG] User {user_id} in matrices: {user_id in state.matrices['storyline_norm'].index}")
    print(f"[DEBUG] User {user_id} rated movies in matrix: {list(state.matrices['storyline_norm'].loc[user_id].dropna().index) if user_id in state.matrices['storyline_norm'].index else 'NOT IN MATRIX'}")
    
    # result is a DataFrame with columns: movie_id, predicted_score, title
    clean_recs = []
    for _, row in result.iterrows():
        clean_recs.append({
            "movie_id":        int(row["movie_id"]),
            "title":           str(row["title"]),
            "predicted_score": round(float(row["predicted_score"]), 4)
        })

    return {
        "user_id":          int(user_id),
        "neighbours_found": len(clean_recs),
        "recommendations":  clean_recs,
        "weight_profile": {
            "w1_storyline":  round(float(weights[0]), 4),
            "w2_acting":     round(float(weights[1]), 4),
            "w3_visuals":    round(float(weights[2]), 4),
            "w4_emotional":  round(float(weights[3]), 4),
            "w5_enjoyment":  round(float(weights[4]), 4),
        }
    }

# ─────────────────────────────────────────────────────────────────────────────
# RATINGS
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/ratings", status_code=201)
def submit_rating(req: RatingRequest, current: dict = Depends(get_current_user)):
    user_id = current["user_id"]
    if state.use_db:
        overall = round((req.storyline + req.acting + req.visuals + req.emotional_impact + req.enjoyment) / 5, 2)
        RatingDB.upsert(
            user_id=user_id,
            movie_id=req.movie_id,
            storyline=req.storyline,
            acting=req.acting,
            visuals=req.visuals,
            emotional_impact=req.emotional_impact,
            enjoyment=req.enjoyment,
            overall_rating=overall
        )
        WeightProfileDB.mark_stale(user_id)
    state.weight_cache.pop(user_id, None)
    # Remove user from matrices so they get re-injected with updated ratings
    if user_id in state.matrices["storyline_norm"].index:
        for matrix in state.matrices.values():
            matrix.drop(index=user_id, inplace=True, errors="ignore")
    return {"message": "Rating submitted", "user_id": user_id,
            "movie_id": req.movie_id}


# ─────────────────────────────────────────────────────────────────────────────
# MOVIES
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/movies")
def get_movies(title: Optional[str] = None, page: int = 1, page_size: int = 20):
    if state.use_db:
        return MovieDB.search(title_query=title, page=page, page_size=page_size)
    _require_dataset()
    movies_df = state.dataset["movies_df"].copy()
    if title:
        movies_df = movies_df[
            movies_df["title"].str.contains(title, case=False, na=False)]
    total  = len(movies_df)
    start  = (page - 1) * page_size
    movies = movies_df.iloc[start:start + page_size].to_dict(orient="records")
    return {"total": total, "page": page, "page_size": page_size, "movies": movies}


@app.get("/movies/{movie_id}")
def get_movie(movie_id: int):
    if state.use_db:
        m = MovieDB.get_by_id(movie_id)
        if not m:
            raise HTTPException(404, f"Movie {movie_id} not found")
        return m
    _require_dataset()
    match = state.dataset["movies_df"]
    match = match[match["movie_id"] == movie_id]
    if match.empty:
        raise HTTPException(404, f"Movie {movie_id} not found")
    return match.iloc[0].to_dict()


# ─────────────────────────────────────────────────────────────────────────────
# USER PROFILE
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/profile/{user_id}")
def get_profile(user_id: int, current: dict = Depends(get_current_user)):
    weights = _get_weights(user_id)
    if state.use_db:
        ratings   = RatingDB.get_user_ratings(user_id)
        n_ratings = len(ratings)
        recent    = ratings[:5]
        profile   = WeightProfileDB.get(user_id)
    elif state.dataset:
        ud        = state.dataset["train_df"]
        ud        = ud[ud["user_id"] == user_id]
        n_ratings = len(ud)
        recent    = ud.tail(5)[["movie_id","overall_rating"]].to_dict("records")
        profile   = None
    else:
        n_ratings, recent, profile = 0, [], None

    return {
        "user_id":        user_id,
        "total_ratings":  n_ratings,
        "weight_profile": {
            "w1_storyline": round(float(weights[0]), 4),
            "w2_acting":    round(float(weights[1]), 4),
            "w3_visuals":   round(float(weights[2]), 4),
            "w4_emotional": round(float(weights[3]), 4),
            "w5_enjoyment": round(float(weights[4]), 4),
        },
        "aga_mae":         profile["best_mae"]    if profile else None,
        "aga_generations": profile["generations"] if profile else None,
        "recent_ratings":  recent
    }


# ─────────────────────────────────────────────────────────────────────────────
# ADMIN
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/admin/stats")
def admin_stats(admin: dict = Depends(require_admin)):
    if state.use_db:
        return {
            "total_users":              len(UserDB.list_all()),
            "total_movies":             MovieDB.total_count(),
            "total_ratings":            RatingDB.total_count(),
            "recommendation_log_count": RecommendationLogDB.total_count(),
            "cached_weight_profiles":   WeightProfileDB.count_cached(),
            "dataset_loaded":           bool(state.dataset),
            "database":                 "connected",
        }
    if state.dataset:
        t = state.dataset["train_df"]
        return {
            "total_users":            int(t["user_id"].nunique()),
            "total_movies":           int(t["movie_id"].nunique()),
            "total_ratings":          int(len(t)),
            "cached_weight_profiles": len(state.weight_cache),
            "dataset_loaded":         True,
            "database":               "offline (demo mode)",
        }
    return {"database": "offline", "dataset_loaded": False}


@app.post("/admin/retrain")
def retrain(user_id: Optional[int] = None, admin: dict = Depends(require_admin)):
    if user_id is not None:
        state.weight_cache.pop(user_id, None)
        if state.use_db:
            WeightProfileDB.mark_stale(user_id)
        return {"message": f"Profile cleared for user {user_id}."}
    else:
        n = len(state.weight_cache)
        state.weight_cache.clear()
        if state.use_db:
            WeightProfileDB.mark_all_stale()
        return {"message": f"All {n} profiles cleared."}


@app.post("/admin/movies", status_code=201)
def add_movie(req: MovieUpsertRequest, admin: dict = Depends(require_admin)):
    if not state.use_db:
        raise HTTPException(503, "Database not available in demo mode")
    MovieDB.upsert(req.movie_id, req.title, req.release_year,
                   req.genres, req.synopsis, req.poster_url)
    return {"message": "Movie saved", "movie_id": req.movie_id}


@app.get("/admin/users")
def list_users(admin: dict = Depends(require_admin)):
    if state.use_db:
        return {"users": UserDB.list_all()}
    return {"users": [], "note": "Database offline"}


@app.delete("/admin/users/{user_id}", status_code=204)
def deactivate_user(user_id: int, admin: dict = Depends(require_admin)):
    if state.use_db:
        UserDB.deactivate(user_id)
    return None
