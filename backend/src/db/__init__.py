"""Database package exposing Base, models, session, and engine management."""

from src.db.base import Base, CrossDialectJson, GUID
from src.db.models import PredictionRecord, ExplanationRecord, utc_now
from src.db.session import (
    get_engine,
    get_sessionmaker,
    get_db,
    ping_database,
    init_db_schema,
    drop_db_schema,
    close_engine,
    get_database_url,
)

__all__ = [
    "Base",
    "CrossDialectJson",
    "GUID",
    "PredictionRecord",
    "ExplanationRecord",
    "utc_now",
    "get_engine",
    "get_sessionmaker",
    "get_db",
    "ping_database",
    "init_db_schema",
    "drop_db_schema",
    "close_engine",
    "get_database_url",
]
