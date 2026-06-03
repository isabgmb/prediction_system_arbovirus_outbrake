"""
core/database.py
----------------
SQLAlchemy async engine, session factory, and declarative base.

Supports both SQLite (local testing) and PostgreSQL (production).
The driver is selected automatically from the DATABASE_URL in .env:
    sqlite+aiosqlite:///./data/sipsa_test.db   → SQLite, no server needed
    postgresql+asyncpg://...                   → PostgreSQL, for production
"""

from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase

from src.core.config import settings

# SQLite needs check_same_thread=False for async use.
# PostgreSQL ignores connect_args entirely, so this is safe for both.
_connect_args = (
    {"check_same_thread": False}
    if settings.database_url.startswith("sqlite")
    else {}
)

# pool_pre_ping not supported by aiosqlite — disable for SQLite
_pool_pre_ping = not settings.database_url.startswith("sqlite")

engine = create_async_engine(
    settings.database_url,
    echo=False,                     # set True to log SQL during development
    connect_args=_connect_args,
    pool_pre_ping=_pool_pre_ping,
)

AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


class Base(DeclarativeBase):
    pass


async def get_db() -> AsyncSession:
    """FastAPI dependency — yields one async DB session per request."""
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
