"""
core/config.py
--------------
Centralised settings loaded from environment variables.
Copy .env.example to .env and fill in your values.
"""

from pydantic_settings import BaseSettings
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent


class Settings(BaseSettings):
    # ── Database ──────────────────────────────────────────────────────────────
    database_url: str = "postgresql+asyncpg://user:password@localhost:5432/dengue"

    # ── Model weights ─────────────────────────────────────────────────────────
    model_weights_path: str = str(ROOT / "data" / "models" / "eval_results.npz")

    # ── Algorithm parameters (must match training in ml/train.py) ─────────────
    n_back_samples:    list[int] = [1, 3, 4, 5, 6]
    train_split:       float     = 0.8
    min_history_weeks: int       = 6       # ValidationService: minimum weeks required

    # ── Auth (RN07) ───────────────────────────────────────────────────────────
    secret_key:                    str = "change-me-in-production"
    algorithm:                     str = "HS256"
    access_token_expire_minutes:   int = 60

    class Config:
        env_file          = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
