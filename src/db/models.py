from contextlib import contextmanager
from datetime import datetime
from typing import Generator, TypeVar

from pydantic import BaseModel
from sqlalchemy import (
    Boolean,
    DateTime,
    String,
    func,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column

from src import create_logger
from src.config import app_settings
from src.config.settings import refresh_settings
from src.db import DatabasePool

logger = create_logger(name="db_utilities")
settings = refresh_settings()
T = TypeVar("T", bound="BaseModel")
D = TypeVar("D", bound="Base")


class Base(DeclarativeBase):
    """Base class for all database models."""

    pass


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
    """Get a database session.

    Yields
    ------
    Session
        A database session
    """
    db_pool = get_db_pool()
    with db_pool.get_session() as session:
        yield session


def init_db() -> None:
    """This function is used to create the tables in the database.
    It should be called once when the application starts."""
    db_pool = get_db_pool()

    # Create all tables in the database
    Base.metadata.create_all(db_pool.engine)
    logger.info("Database initialized")


# ===== Database Models =====
class Users(Base):
    """Data model for storing user information."""

    __tablename__: str = "users"
    id: Mapped[int] = mapped_column("id", primary_key=True)
    username: Mapped[str] = mapped_column(String(50), unique=True, nullable=False)
    full_name: Mapped[str] = mapped_column("fullName", String(255), nullable=False)
    email: Mapped[str] = mapped_column(String(255), unique=True, nullable=False)
    hashed_password: Mapped[str] = mapped_column(
        "hashedPassword", String(255), nullable=False
    )
    disabled: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=func.now()
    )

    def __repr__(self) -> str:
        """
        Returns a string representation of the Users object.

        Returns
        -------
        str
        """
        return f"{self.__class__.__name__}(id={self.id!r}, username={self.username!r}, email={self.email!r})"
