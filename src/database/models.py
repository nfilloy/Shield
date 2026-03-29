"""
SQLAlchemy models for SHIELD user management and analysis history.

Defines the database schema for:
- User: Application users with authentication
- Analysis: History of phishing/smishing analyses
"""

from datetime import datetime
from typing import Optional, List
from sqlalchemy import (
    String, Integer, Float, Text, DateTime, ForeignKey, Index
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    """Base class for all SQLAlchemy models."""
    pass


class User(Base):
    """
    User model for authentication and authorization.

    Attributes:
        id: Primary key
        username: Unique username for login
        email: Unique email address
        password_hash: Hashed password (never store plain text)
        role: User role ('user' or 'admin')
        created_at: Account creation timestamp
        last_login: Last successful login timestamp
        analyses: Relationship to user's analysis history
    """
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    username: Mapped[str] = mapped_column(String(50), unique=True, nullable=False, index=True)
    email: Mapped[str] = mapped_column(String(255), unique=True, nullable=False, index=True)
    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    role: Mapped[str] = mapped_column(String(20), nullable=False, default="user")
    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        nullable=False,
        default=datetime.utcnow
    )
    last_login: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)

    # Relationship: One user -> Many analyses
    analyses: Mapped[List["Analysis"]] = relationship(
        "Analysis",
        back_populates="user",
        cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<User(id={self.id}, username='{self.username}', role='{self.role}')>"


class Analysis(Base):
    """
    Analysis model for storing phishing/smishing detection history.

    Attributes:
        id: Primary key
        user_id: Foreign key to User (nullable for anonymous analyses)
        text_input: Original text analyzed
        text_type: Type of content ('sms' or 'email')
        model_used: Name of ML model used for prediction
        prediction: Binary prediction (0=safe, 1=threat)
        probability: Threat probability percentage (0-100)
        features_json: Extracted features stored as JSON string
        created_at: Analysis timestamp
        user: Relationship back to User
    """
    __tablename__ = "analyses"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[Optional[int]] = mapped_column(
        Integer,
        ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True,
        index=True
    )
    text_input: Mapped[str] = mapped_column(Text, nullable=False)
    text_type: Mapped[str] = mapped_column(String(20), nullable=False)  # 'sms' or 'email'
    model_used: Mapped[str] = mapped_column(String(50), nullable=False)
    prediction: Mapped[int] = mapped_column(Integer, nullable=False)  # 0 or 1
    probability: Mapped[float] = mapped_column(Float, nullable=False)  # 0.0 to 100.0
    features_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)  # JSON string
    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        nullable=False,
        default=datetime.utcnow,
        index=True
    )

    # Relationship: Many analyses -> One user
    user: Mapped[Optional["User"]] = relationship("User", back_populates="analyses")

    # Composite index for common queries
    __table_args__ = (
        Index('idx_analysis_user_created', 'user_id', 'created_at'),
        Index('idx_analysis_type_created', 'text_type', 'created_at'),
    )

    def __repr__(self) -> str:
        verdict = "THREAT" if self.prediction == 1 else "SAFE"
        return f"<Analysis(id={self.id}, type='{self.text_type}', verdict={verdict}, prob={self.probability:.1f}%)>"
