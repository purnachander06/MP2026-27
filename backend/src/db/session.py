"""Asynchronous database session management, connection pooling, and health ping with smart failover."""

import asyncio
import logging
import os
from typing import AsyncGenerator, Optional, Tuple
from sqlalchemy.ext.asyncio import (
    create_async_engine,
    async_sessionmaker,
    AsyncSession,
    AsyncEngine,
)
from sqlalchemy import text
from src.db.base import Base

logger = logging.getLogger("DatabaseSession")

SUPABASE_FALLBACK_URL = "postgresql+asyncpg://postgres.mxiyhuombyyfjytvwfgk:Sup%40base112@aws-0-ap-southeast-2.pooler.supabase.com:5432/postgres?ssl=require"
SQLITE_FALLBACK_URL = "sqlite+aiosqlite:///./iot_security.db"

_engine: Optional[AsyncEngine] = None
_sessionmaker: Optional[async_sessionmaker[AsyncSession]] = None
_active_db_url: Optional[str] = None
_fallback_lock = asyncio.Lock()


def get_database_url() -> str:
    return get_configured_database_url()

def get_configured_database_url() -> str:
    """Retrieve database URL from environment or default to Supabase."""
    url = os.getenv("DATABASE_URL", "").strip()
    if not url:
        return SUPABASE_FALLBACK_URL
    return url


def _create_engine_for_url(db_url: str) -> AsyncEngine:
    is_sqlite = db_url.startswith("sqlite")
    engine_kwargs = {
        "echo": os.getenv("DATABASE_ECHO", "false").lower() in ("true", "1"),
        "future": True,
    }
    if not is_sqlite:
        engine_kwargs.update({
            "pool_size": int(os.getenv("DATABASE_POOL_SIZE", 5)),
            "max_overflow": int(os.getenv("DATABASE_MAX_OVERFLOW", 5)),
            "pool_timeout": float(os.getenv("DATABASE_POOL_TIMEOUT", 15.0)),
            "pool_pre_ping": True,
        })
    else:
        engine_kwargs["connect_args"] = {"check_same_thread": False}

    safe_url = db_url
    if "@" in safe_url and "://" in safe_url:
        parts = safe_url.split("@")
        scheme = parts[0].split("://")[0]
        safe_url = f"{scheme}://***:***@{parts[1]}"

    logger.info(f"Initializing database engine for: {safe_url}")
    return create_async_engine(db_url, **engine_kwargs)


def get_engine() -> AsyncEngine:
    global _engine, _active_db_url
    if _engine is None:
        db_url = get_configured_database_url()
        _active_db_url = db_url
        _engine = _create_engine_for_url(db_url)
    return _engine


def get_sessionmaker() -> async_sessionmaker[AsyncSession]:
    global _sessionmaker
    if _sessionmaker is None:
        engine = get_engine()
        _sessionmaker = async_sessionmaker(
            bind=engine,
            class_=AsyncSession,
            expire_on_commit=False,
            autoflush=False,
        )
    return _sessionmaker


async def switch_to_fallback_database(reason: str = ""):
    """Switch from an unreachable database to Supabase or SQLite."""
    global _engine, _sessionmaker, _active_db_url
    async with _fallback_lock:
        logger.warning(f"Database connection issue ({reason}). Activating resilient fallback...")

        if _engine is not None:
            try:
                await _engine.dispose()
            except Exception:
                pass

        # If we were attempting localhost, try Supabase first
        if _active_db_url != SUPABASE_FALLBACK_URL:
            try:
                logger.info("Connecting to Supabase cloud database (AWS Sydney pooler)...")
                new_engine = _create_engine_for_url(SUPABASE_FALLBACK_URL)
                async with new_engine.connect() as conn:
                    await conn.execute(text("SELECT 1"))
                _engine = new_engine
                _active_db_url = SUPABASE_FALLBACK_URL
                _sessionmaker = async_sessionmaker(
                    bind=_engine,
                    class_=AsyncSession,
                    expire_on_commit=False,
                    autoflush=False,
                )
                logger.info("Successfully connected to Supabase cloud database as active engine.")
                return
            except Exception as e:
                logger.warning(f"Supabase fallback connection failed: {e}")

        # Final fallback: local SQLite
        logger.info("Falling back to local SQLite database (iot_security.db)...")
        new_engine = _create_engine_for_url(SQLITE_FALLBACK_URL)
        async with new_engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        _engine = new_engine
        _active_db_url = SQLITE_FALLBACK_URL
        _sessionmaker = async_sessionmaker(
            bind=_engine,
            class_=AsyncSession,
            expire_on_commit=False,
            autoflush=False,
        )
        logger.info("Local SQLite database initialized and active.")


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency yielding an isolated, request-scoped AsyncSession with failover."""
    session_factory = get_sessionmaker()
    try:
        async with session_factory() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise
            finally:
                await session.close()
    except Exception as conn_err:
        err_str = str(conn_err)
        if any(kw in err_str.lower() for kw in ("password", "connection", "refused", "reset", "closed in the middle", "does not exist")):
            await switch_to_fallback_database(reason=err_str)
            fallback_factory = get_sessionmaker()
            async with fallback_factory() as fallback_session:
                try:
                    yield fallback_session
                    await fallback_session.commit()
                except Exception:
                    await fallback_session.rollback()
                    raise
                finally:
                    await fallback_session.close()
        else:
            raise


async def ping_database() -> Tuple[bool, Optional[str]]:
    """Verify database connectivity, with automatic fallback if connection fails."""
    try:
        engine = get_engine()
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        return True, None
    except Exception as e:
        error_msg = f"{type(e).__name__}: {str(e)}"
        logger.warning(f"Database health ping failed on initial engine: {error_msg}")
        await switch_to_fallback_database(reason=error_msg)
        try:
            fallback_engine = get_engine()
            async with fallback_engine.connect() as conn:
                await conn.execute(text("SELECT 1"))
            return True, None
        except Exception as e2:
            return False, str(e2)


async def init_db_schema() -> None:
    engine = get_engine()
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def drop_db_schema() -> None:
    engine = get_engine()
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)


async def close_engine() -> None:
    global _engine, _sessionmaker
    if _engine is not None:
        await _engine.dispose()
        _engine = None
        _sessionmaker = None