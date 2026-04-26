"""
database.py
===========
MySQL database connection layer for the MCRS.

Provides:
  - get_connection()        — context-managed PyMySQL connection
  - init_db()               — create all tables (runs schema.sql)
  - UserDB                  — user CRUD operations
  - MovieDB                 — movie CRUD operations
  - RatingDB                — rating CRUD operations
  - WeightProfileDB         — weight profile read/write/cache operations
  - RecommendationLogDB     — recommendation event logging

All functions use parameterised queries to prevent SQL injection.
"""

from __future__ import annotations


import pymysql
import pymysql.cursors
import json
import os
from contextlib import contextmanager
from datetime import datetime
from typing import Optional

from config import settings


# ─────────────────────────────────────────────────────────────────────────────
# CONNECTION
# ─────────────────────────────────────────────────────────────────────────────

@contextmanager
def get_connection():
    """
    Context manager that yields a PyMySQL connection with DictCursor.
    Commits on success, rolls back on exception, always closes.

    Usage:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT ...")
    """
    conn = pymysql.connect(
        host=settings.db_host,
        port=settings.db_port,
        db=settings.db_name,
        user=settings.db_user,
        password=settings.db_password,
        charset="utf8mb4",
        cursorclass=pymysql.cursors.DictCursor,
        autocommit=False,
    )
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def test_connection() -> bool:
    """Return True if DB is reachable, False otherwise."""
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
        return True
    except Exception as e:
        print(f"[database] Connection failed: {e}")
        return False


# ─────────────────────────────────────────────────────────────────────────────
# INIT — CREATE TABLES
# ─────────────────────────────────────────────────────────────────────────────

def init_db():
    """
    Create all MCRS tables if they do not already exist.
    Reads schema.sql and executes each statement.
    """
    schema_path = os.path.join(os.path.dirname(__file__), "schema.sql")
    if not os.path.exists(schema_path):
        print("[database] schema.sql not found — skipping table creation.")
        return

    with open(schema_path, encoding="utf-8") as f:
        sql = f.read()

    # Split on semicolons and execute non-empty statements
    statements = [s.strip() for s in sql.split(";") if s.strip()]

    with get_connection() as conn:
        with conn.cursor() as cur:
            for stmt in statements:
                if stmt.upper().startswith(("CREATE", "USE", "INSERT",
                                            "ALTER", "DROP", "SELECT")):
                    try:
                        cur.execute(stmt)
                    except Exception as e:
                        print(f"[database] init_db warning: {e}")
    print("[database] Tables initialised successfully.")


# ─────────────────────────────────────────────────────────────────────────────
# USERS
# ─────────────────────────────────────────────────────────────────────────────

class UserDB:

    @staticmethod
    def create(username: str, email: str,
               password_hash: str, role: str = "user") -> int:
        """
        Insert a new user. Returns the new user_id.
        Raises pymysql.IntegrityError if username or email already exists.
        """
        sql = """
            INSERT INTO Users (username, email, password_hash, role)
            VALUES (%s, %s, %s, %s)
        """
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (username, email, password_hash, role))
                return cur.lastrowid

    @staticmethod
    def get_by_username(username: str) -> Optional[dict]:
        """Return user dict or None."""
        sql = "SELECT * FROM Users WHERE username = %s LIMIT 1"
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (username,))
                return cur.fetchone()

    @staticmethod
    def get_by_id(user_id: int) -> Optional[dict]:
        """Return user dict or None."""
        sql = "SELECT * FROM Users WHERE user_id = %s LIMIT 1"
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (user_id,))
                return cur.fetchone()

    @staticmethod
    def update_last_login(user_id: int):
        """Update last_login timestamp to now."""
        sql = "UPDATE Users SET last_login = %s WHERE user_id = %s"
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (datetime.utcnow(), user_id))

    @staticmethod
    def list_all() -> list[dict]:
        """Return all users (admin use)."""
        sql = "SELECT user_id, username, email, role, is_active, created_at FROM Users"
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql)
                return cur.fetchall()

    @staticmethod
    def deactivate(user_id: int):
        sql = "UPDATE Users SET is_active = FALSE WHERE user_id = %s"
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (user_id,))


# ─────────────────────────────────────────────────────────────────────────────
# MOVIES
# ─────────────────────────────────────────────────────────────────────────────

class MovieDB:

    @staticmethod
    def bulk_insert(movies: list[dict]):
        """
        Insert many movies at once (used during dataset loading).
        movies: list of {movie_id, title, release_year?, genres?}
        Skips duplicates via INSERT IGNORE.
        """
        sql = """
            INSERT IGNORE INTO Movies (movie_id, title, release_year, genres)
            VALUES (%s, %s, %s, %s)
        """
        rows = [
            (m["movie_id"], m["title"],
             m.get("release_year"), m.get("genres"))
            for m in movies
        ]
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.executemany(sql, rows)
        print(f"[database] Inserted/skipped {len(rows)} movies.")

    @staticmethod
    def get_by_id(movie_id: int) -> Optional[dict]:
        sql = "SELECT * FROM Movies WHERE movie_id = %s LIMIT 1"
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (movie_id,))
                return cur.fetchone()

    @staticmethod
    def search(title_query: str = None,
               page: int = 1,
               page_size: int = 20) -> dict:
        """
        Search movies by title (partial match) with pagination.
        Returns {total, page, page_size, movies: [...]}.
        """
        offset = (page - 1) * page_size

        if title_query:
            count_sql = ("SELECT COUNT(*) as cnt FROM Movies "
                         "WHERE title LIKE %s")
            fetch_sql = ("SELECT * FROM Movies WHERE title LIKE %s "
                         "ORDER BY movie_id LIMIT %s OFFSET %s")
            pattern = f"%{title_query}%"
            params_count = (pattern,)
            params_fetch = (pattern, page_size, offset)
        else:
            count_sql = "SELECT COUNT(*) as cnt FROM Movies"
            fetch_sql  = "SELECT * FROM Movies ORDER BY movie_id LIMIT %s OFFSET %s"
            params_count = ()
            params_fetch  = (page_size, offset)

        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(count_sql, params_count)
                total = cur.fetchone()["cnt"]
                cur.execute(fetch_sql, params_fetch)
                movies = cur.fetchall()

        return {"total": total, "page": page,
                "page_size": page_size, "movies": movies}

    @staticmethod
    def upsert(movie_id: int, title: str,
               release_year: int = None,
               genres: str = None,
               synopsis: str = None,
               poster_url: str = None):
        sql = """
            INSERT INTO Movies (movie_id, title, release_year, genres,
                                synopsis, poster_url)
            VALUES (%s, %s, %s, %s, %s, %s)
            ON DUPLICATE KEY UPDATE
                title        = VALUES(title),
                release_year = VALUES(release_year),
                genres       = VALUES(genres),
                synopsis     = VALUES(synopsis),
                poster_url   = VALUES(poster_url)
        """
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (movie_id, title, release_year,
                                  genres, synopsis, poster_url))

    @staticmethod
    def total_count() -> int:
        sql = "SELECT COUNT(*) as cnt FROM Movies"
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql)
                return cur.fetchone()["cnt"]


# ─────────────────────────────────────────────────────────────────────────────
# RATINGS
# ─────────────────────────────────────────────────────────────────────────────

class RatingDB:

    @staticmethod
    def upsert(user_id: int, movie_id: int,
               storyline: float, acting: float, visuals: float,
               emotional_impact: float, enjoyment: float,
               overall_rating: float):
        """
        Insert or update a rating record.
        If (user_id, movie_id) already exists, update all criterion scores.
        """
        sql = """
            INSERT INTO Ratings
                (user_id, movie_id, storyline, acting, visuals,
                 emotional_impact, enjoyment, overall_rating, timestamp)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON DUPLICATE KEY UPDATE
                storyline        = VALUES(storyline),
                acting           = VALUES(acting),
                visuals          = VALUES(visuals),
                emotional_impact = VALUES(emotional_impact),
                enjoyment        = VALUES(enjoyment),
                overall_rating   = VALUES(overall_rating),
                timestamp        = VALUES(timestamp)
        """
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (
                    user_id, movie_id,
                    round(storyline, 2), round(acting, 2),
                    round(visuals, 2), round(emotional_impact, 2),
                    round(enjoyment, 2), round(overall_rating, 2),
                    datetime.utcnow()
                ))

    @staticmethod
    def bulk_insert(ratings: list[dict]):
        """
        Insert many ratings at once (used during dataset loading).
        Skips duplicates. ratings: list of dicts with all rating fields.
        """
        sql = """
            INSERT IGNORE INTO Ratings
                (user_id, movie_id, storyline, acting, visuals,
                 emotional_impact, enjoyment, overall_rating, timestamp)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        rows = [(
            r["user_id"], r["movie_id"],
            round(float(r["storyline"]), 2),
            round(float(r["acting"]), 2),
            round(float(r["visuals"]), 2),
            round(float(r["emotional_impact"]), 2),
            round(float(r["enjoyment"]), 2),
            round(float(r["overall_rating"]), 2),
            datetime.utcnow()
        ) for r in ratings]

        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.executemany(sql, rows)
        print(f"[database] Inserted/skipped {len(rows)} ratings.")

    @staticmethod
    def get_user_ratings(user_id: int) -> list[dict]:
        """Return all ratings submitted by a given user."""
        sql = """
            SELECT r.*, m.title
            FROM Ratings r
            LEFT JOIN Movies m ON r.movie_id = m.movie_id
            WHERE r.user_id = %s
            ORDER BY r.timestamp DESC
        """
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (user_id,))
                return cur.fetchall()

    @staticmethod
    def get_user_training_ratings(user_id: int) -> list[dict]:
        """
        Return all ratings for a user as raw dicts suitable for AGA training.
        Includes normalised columns computed on the fly.
        """
        sql = """
            SELECT
                user_id, movie_id,
                storyline, acting, visuals, emotional_impact, enjoyment,
                overall_rating,
                (storyline        - 1) / 4.0 AS storyline_norm,
                (acting           - 1) / 4.0 AS acting_norm,
                (visuals          - 1) / 4.0 AS visuals_norm,
                (emotional_impact - 1) / 4.0 AS emotional_impact_norm,
                (enjoyment        - 1) / 4.0 AS enjoyment_norm,
                (overall_rating   - 1) / 4.0 AS overall_norm
            FROM Ratings
            WHERE user_id = %s
        """
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (user_id,))
                return cur.fetchall()

    @staticmethod
    def get_all_ratings_for_matrix() -> list[dict]:
        """
        Return all ratings with normalised scores for building
        the user-item matrices used by the CF engine.
        """
        sql = """
            SELECT
                user_id, movie_id, overall_rating,
                (storyline        - 1) / 4.0 AS storyline_norm,
                (acting           - 1) / 4.0 AS acting_norm,
                (visuals          - 1) / 4.0 AS visuals_norm,
                (emotional_impact - 1) / 4.0 AS emotional_impact_norm,
                (enjoyment        - 1) / 4.0 AS enjoyment_norm,
                (overall_rating   - 1) / 4.0 AS overall_norm
            FROM Ratings
        """
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql)
                return cur.fetchall()

    @staticmethod
    def count_user_ratings(user_id: int) -> int:
        sql = "SELECT COUNT(*) as cnt FROM Ratings WHERE user_id = %s"
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (user_id,))
                return cur.fetchone()["cnt"]

    @staticmethod
    def total_count() -> int:
        sql = "SELECT COUNT(*) as cnt FROM Ratings"
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql)
                return cur.fetchone()["cnt"]


# ─────────────────────────────────────────────────────────────────────────────
# WEIGHT PROFILES
# ─────────────────────────────────────────────────────────────────────────────

class WeightProfileDB:

    @staticmethod
    def get(user_id: int) -> Optional[dict]:
        """Return the weight profile for a user, or None if not found."""
        sql = """
            SELECT * FROM WeightProfiles
            WHERE user_id = %s AND is_stale = FALSE
            LIMIT 1
        """
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (user_id,))
                return cur.fetchone()

    @staticmethod
    def upsert(user_id: int,
               w1: float, w2: float, w3: float, w4: float, w5: float,
               best_mae: float, generations: int, converged: bool):
        """
        Insert or update the weight profile for a user.
        Marks profile as fresh (is_stale = FALSE).
        """
        sql = """
            INSERT INTO WeightProfiles
                (user_id, w1_storyline, w2_acting, w3_visuals,
                 w4_emotional, w5_enjoyment, best_mae, generations,
                 converged, computed_at, is_stale)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, FALSE)
            ON DUPLICATE KEY UPDATE
                w1_storyline = VALUES(w1_storyline),
                w2_acting    = VALUES(w2_acting),
                w3_visuals   = VALUES(w3_visuals),
                w4_emotional = VALUES(w4_emotional),
                w5_enjoyment = VALUES(w5_enjoyment),
                best_mae     = VALUES(best_mae),
                generations  = VALUES(generations),
                converged    = VALUES(converged),
                computed_at  = VALUES(computed_at),
                is_stale     = FALSE
        """
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (
                    user_id,
                    round(w1, 4), round(w2, 4), round(w3, 4),
                    round(w4, 4), round(w5, 4),
                    round(best_mae, 6), generations, converged,
                    datetime.utcnow()
                ))

    @staticmethod
    def mark_stale(user_id: int):
        """
        Mark a user's weight profile as stale.
        Called whenever the user submits a new rating — forces AGA re-run
        on next recommendation request.
        """
        sql = """
            UPDATE WeightProfiles
            SET is_stale = TRUE
            WHERE user_id = %s
        """
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (user_id,))

    @staticmethod
    def mark_all_stale():
        """Mark ALL weight profiles as stale (admin: force global retrain)."""
        sql = "UPDATE WeightProfiles SET is_stale = TRUE"
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql)

    @staticmethod
    def count_cached() -> int:
        """Return number of fresh (non-stale) weight profiles."""
        sql = ("SELECT COUNT(*) as cnt FROM WeightProfiles "
               "WHERE is_stale = FALSE")
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql)
                return cur.fetchone()["cnt"]


# ─────────────────────────────────────────────────────────────────────────────
# RECOMMENDATION LOG
# ─────────────────────────────────────────────────────────────────────────────

class RecommendationLogDB:

    @staticmethod
    def log(user_id: int,
            recommended_ids: list[int],
            predicted_scores: list[float],
            weights_used: dict = None):
        """
        Record a recommendation generation event.
        """
        sql = """
            INSERT INTO RecommendationLog
                (user_id, recommended_ids, predicted_scores,
                 generated_at, weights_used)
            VALUES (%s, %s, %s, %s, %s)
        """
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (
                    user_id,
                    json.dumps(recommended_ids),
                    json.dumps([round(s, 4) for s in predicted_scores]),
                    datetime.utcnow(),
                    json.dumps(weights_used) if weights_used else None
                ))

    @staticmethod
    def get_user_history(user_id: int, limit: int = 10) -> list[dict]:
        """Return the N most recent recommendation events for a user."""
        sql = """
            SELECT * FROM RecommendationLog
            WHERE user_id = %s
            ORDER BY generated_at DESC
            LIMIT %s
        """
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (user_id, limit))
                rows = cur.fetchall()
        # Parse JSON fields
        for row in rows:
            if isinstance(row.get("recommended_ids"), str):
                row["recommended_ids"] = json.loads(row["recommended_ids"])
            if isinstance(row.get("predicted_scores"), str):
                row["predicted_scores"] = json.loads(row["predicted_scores"])
        return rows

    @staticmethod
    def total_count() -> int:
        sql = "SELECT COUNT(*) as cnt FROM RecommendationLog"
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql)
                return cur.fetchone()["cnt"]
