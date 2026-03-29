"""
Database module for SHIELD application.

Provides SQLAlchemy models and connection utilities for:
- User management and authentication
- Analysis history tracking

Usage:
    from src.database import init_db, get_db_session, User, Analysis

    # Initialize database (create tables)
    init_db()

    # Use context manager for sessions
    with get_db_session() as session:
        user = User(username="admin", email="admin@shield.io", password_hash="...")
        session.add(user)
"""

from .models import Base, User, Analysis
from .connection import (
    init_db,
    get_session,
    get_db_session,
    get_database_path,
    get_engine,
    drop_all_tables
)
from .analysis_service import (
    save_analysis,
    get_user_analyses,
    get_analysis_by_id,
    get_user_analysis_count,
    get_user_stats,
    delete_analysis,
    get_all_analyses_count,
    get_global_stats,
    get_recent_analyses
)

__all__ = [
    # Models
    "Base",
    "User",
    "Analysis",
    # Connection utilities
    "init_db",
    "get_session",
    "get_db_session",
    "get_database_path",
    "get_engine",
    "drop_all_tables",
    # Analysis service
    "save_analysis",
    "get_user_analyses",
    "get_analysis_by_id",
    "get_user_analysis_count",
    "get_user_stats",
    "delete_analysis",
    "get_all_analyses_count",
    "get_global_stats",
    "get_recent_analyses",
]
