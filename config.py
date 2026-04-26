"""
config.py
=========
Central configuration for the MCRS backend.
All sensitive values are loaded from environment variables or a .env file.

Usage in other modules:
    from config import settings
    print(settings.db_host)
"""

from __future__ import annotations


import os
from dataclasses import dataclass, field


@dataclass
class Settings:
    # ── Database ──────────────────────────────────────────────────────────────
    db_host:     str = os.getenv("DB_HOST",     "localhost")
    db_port:     int = int(os.getenv("DB_PORT", "3306"))
    db_name:     str = os.getenv("DB_NAME",     "mcrs_db")
    db_user:     str = os.getenv("DB_USER",     "root")
    db_password: str = os.getenv("DB_PASSWORD", "J0sh_uqq@")

    # ── JWT Auth ──────────────────────────────────────────────────────────────
    secret_key:       str = os.getenv("SECRET_KEY", "mcrs-secret-key-change-in-production")
    algorithm:        str = "HS256"
    token_expire_min: int = int(os.getenv("TOKEN_EXPIRE_MIN", "60"))

    # ── Dataset ───────────────────────────────────────────────────────────────
    movielens_dir:     str = os.getenv("MOVIELENS_DIR",     "data/ml-100k/ml-100k")
    movielens_version: str = os.getenv("MOVIELENS_VERSION", "100k")

    # ── AGA / Recommendation ─────────────────────────────────────────────────
    neighbourhood_k: int   = int(os.getenv("NEIGHBOURHOOD_K",   "30"))
    top_n:           int   = int(os.getenv("TOP_N",             "10"))
    weight_profiles: str   = os.getenv("WEIGHT_PROFILES_FILE", "weight_profiles.json")

    # ── CORS ──────────────────────────────────────────────────────────────────
    allowed_origins: list = field(default_factory=lambda: [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ])

    @property
    def db_url(self) -> str:
        return (f"mysql+pymysql://{self.db_user}:{self.db_password}"
                f"@{self.db_host}:{self.db_port}/{self.db_name}")


# Singleton instance used across the application
settings = Settings()


def load_dotenv(path=".env"):
    """
    Simple .env loader — reads KEY=VALUE lines and sets os.environ.
    Call this before importing settings if you use a .env file.
    """
    if not os.path.exists(path):
        return
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip())
