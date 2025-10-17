from contextlib import contextmanager
from datetime import datetime
from typing import Generator

from sqlalchemy import Boolean, Column, DateTime, ForeignKey, String, Table, func
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, relationship

from src.config import app_settings
from src.config.settings import refresh_settings
from src.db import DatabasePool

settings = refresh_settings()


class Base(DeclarativeBase):
    """Base class for all database models."""

    pass


# =========================================================
# ==================== Database Models ====================
# =========================================================
# Association table for many-to-many relationship between users and roles
user_roles = Table(
    "user_roles",
    Base.metadata,
    Column("user_id", ForeignKey("users.id"), primary_key=True),
    Column("role_id", ForeignKey("roles.id"), primary_key=True),
)


class DBUser(Base):
    """Data model for storing user information."""

    __tablename__: str = "users"
    id: Mapped[int] = mapped_column("id", primary_key=True)
    username: Mapped[str] = mapped_column(String(50), unique=True, nullable=False)
    full_name: Mapped[str] = mapped_column("fullName", String(255), nullable=False)
    email: Mapped[str] = mapped_column(String(255), unique=True, nullable=False)
    hashed_password: Mapped[str] = mapped_column(
        "hashedPassword", String(255), nullable=False
    )
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=func.now()
    )

    # Many-to-many relationship with roles
    roles = relationship("DBRole", secondary=user_roles, back_populates="users")

    def __repr__(self) -> str:
        """
        Returns a string representation of the Users object.

        Returns
        -------
        str
        """
        return f"{self.__class__.__name__}(id={self.id!r}, username={self.username!r}, email={self.email!r})"


class DBRole(Base):
    """Data model for storing user roles."""

    __tablename__: str = "roles"

    id: Mapped[int] = mapped_column("id", primary_key=True)
    name: Mapped[str] = mapped_column(String(50), unique=True, nullable=False)
    description: Mapped[str] = mapped_column(String(255), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=func.now()
    )

    # Many-to-many relationship with users
    users = relationship("DBUser", secondary=user_roles, back_populates="roles")

    def __repr__(self) -> str:
        """
        Returns a string representation of the DBRole object.

        Returns
        -------
        str
        """
        return f"{self.__class__.__name__}(id={self.id!r}, name={self.name!r})"


# =========================================================
# ==================== Utilities ==========================
# =========================================================
# Global pool instance
_db_pool: DatabasePool | None = None


def get_db_pool() -> DatabasePool:
    """Get or create the global database pool."""
    global _db_pool
    if _db_pool is None:
        _db_pool = DatabasePool(app_settings.database_url)
    return _db_pool


@contextmanager
def get_db_session() -> Generator[Session, None, None]:
    """Get a database session context manager.

    Use this for manual session management with 'with' statements.

    Yields
    ------
    Session
        A database session
    """
    db_pool = get_db_pool()
    with db_pool.get_session() as session:
        yield session


def get_db() -> Generator[Session, None, None]:
    """FastAPI dependency for database sessions.

    This is a generator function that FastAPI will handle automatically.
    Use this with Depends() in your route handlers.

    Yields
    ------
    Session
        A database session that will be automatically closed after the request
    """
    db_pool = get_db_pool()
    with db_pool.get_session() as session:
        try:
            yield session
        finally:
            # Session cleanup is handled by the context manager
            pass
