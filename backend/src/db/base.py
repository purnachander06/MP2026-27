"""Base Declarative ORM and cross-dialect type definitions."""

import uuid
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.types import TypeDecorator, JSON, String
from sqlalchemy.dialects.postgresql import JSONB, UUID as PG_UUID


class Base(DeclarativeBase):
    """Base class for all SQLAlchemy ORM models."""
    pass


class CrossDialectJson(TypeDecorator):
    """
    Cross-dialect JSON type decorator.
    Uses native PostgreSQL JSONB on PostgreSQL, and standard JSON on SQLite/other dialects.
    """
    impl = JSON
    cache_ok = True

    def load_dialect_impl(self, dialect):
        if dialect.name == "postgresql":
            return dialect.type_descriptor(JSONB())
        return dialect.type_descriptor(JSON())


class GUID(TypeDecorator):
    """
    Platform-independent GUID type.
    Uses PostgreSQL UUID on PostgreSQL, and CHAR(36) / String(36) on SQLite.
    """
    impl = String(36)
    cache_ok = True

    def load_dialect_impl(self, dialect):
        if dialect.name == "postgresql":
            return dialect.type_descriptor(PG_UUID(as_uuid=True))
        return dialect.type_descriptor(String(36))

    def process_bind_param(self, value, dialect):
        if value is None:
            return value
        elif dialect.name == "postgresql":
            return str(value)
        else:
            if not isinstance(value, uuid.UUID):
                return str(uuid.UUID(str(value)))
            return str(value)

    def process_result_value(self, value, dialect):
        if value is None:
            return value
        return str(value)
