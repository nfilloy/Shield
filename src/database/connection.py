"""
Database connection management for SHIELD application.

Provides functions for:
- Database initialization (creating tables)
- Session management (getting database connections)
- Database path configuration
"""

import logging
from pathlib import Path
from contextlib import contextmanager
from typing import Generator

from sqlalchemy import create_engine, event
from sqlalchemy.orm import Session, sessionmaker

from .models import Base

logger = logging.getLogger(__name__)

# Database configuration
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATABASE_PATH = PROJECT_ROOT / "data" / "shield.db"
DATABASE_URL = f"sqlite:///{DATABASE_PATH}"

# Engine and session factory (created lazily)
_engine = None
_SessionLocal = None


def get_engine():
    """
    Get or create the SQLAlchemy engine.

    Uses lazy initialization to avoid creating the engine until needed.
    Enables SQLite foreign key support.

    Returns:
        SQLAlchemy Engine instance
    """
    global _engine

    if _engine is None:
        # Ensure data directory exists
        DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)

        _engine = create_engine(
            DATABASE_URL,
            echo=False,  # Set to True for SQL debugging
            connect_args={"check_same_thread": False}  # Needed for SQLite + threading
        )

        # Enable foreign key support in SQLite
        @event.listens_for(_engine, "connect")
        def set_sqlite_pragma(dbapi_connection, connection_record):
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.close()

        logger.info(f"Database engine created: {DATABASE_PATH}")

    return _engine


def get_session_factory():
    """
    Get or create the session factory.

    Returns:
        SQLAlchemy sessionmaker instance
    """
    global _SessionLocal

    if _SessionLocal is None:
        _SessionLocal = sessionmaker(
            autocommit=False,
            autoflush=False,
            bind=get_engine()
        )

    return _SessionLocal


def init_db() -> bool:
    """
    Initialize the database by creating all tables.

    This function is idempotent - it can be called multiple times safely.
    Tables are only created if they don't already exist.

    Returns:
        True if successful, False otherwise
    """
    try:
        engine = get_engine()
        Base.metadata.create_all(bind=engine)
        logger.info(f"Database initialized successfully at: {DATABASE_PATH}")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        return False


def get_session() -> Session:
    """
    Create a new database session.

    The caller is responsible for closing the session when done.
    For most use cases, prefer using get_db_session() context manager.

    Returns:
        New SQLAlchemy Session instance
    """
    SessionLocal = get_session_factory()
    return SessionLocal()


@contextmanager
def get_db_session() -> Generator[Session, None, None]:
    """
    Context manager for database sessions.

    Automatically handles session lifecycle:
    - Creates a new session
    - Commits on success
    - Rolls back on exception
    - Closes session when done

    Usage:
        with get_db_session() as session:
            user = User(username="test", ...)
            session.add(user)

    Yields:
        SQLAlchemy Session instance
    """
    session = get_session()
    try:
        yield session
        session.commit()
    except Exception as e:
        session.rollback()
        logger.error(f"Database session error: {e}")
        raise
    finally:
        session.close()


def drop_all_tables() -> bool:
    """
    Drop all tables from the database.

    WARNING: This will delete all data! Use only for testing/development.

    Returns:
        True if successful, False otherwise
    """
    try:
        engine = get_engine()
        Base.metadata.drop_all(bind=engine)
        logger.warning("All database tables dropped!")
        return True
    except Exception as e:
        logger.error(f"Failed to drop tables: {e}")
        return False


def get_database_path() -> Path:
    """
    Get the path to the SQLite database file.

    Returns:
        Path object pointing to shield.db
    """
    return DATABASE_PATH
