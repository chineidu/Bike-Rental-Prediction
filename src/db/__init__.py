from contextlib import contextmanager
from typing import Generator

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import QueuePool

from src import create_logger
from src.config.settings import refresh_settings

settings = refresh_settings()
logger = create_logger(name="db_utilities")


class DatabasePool:
    """Database connection pool with automatic reconnection."""

    def __init__(self, database_url: str) -> None:
        """Initialize"""
        self.database_url: str = database_url
        self._engine: Engine | None = None
        self._session_factory: sessionmaker[Session] | None = None
        self._setup_engine()

    def _setup_engine(self) -> None:
        """Set up the database engine and session factory."""
        self._engine = create_engine(
            self.database_url,
            poolclass=QueuePool,
            pool_size=10,  # Keep 10 connections in pool
            max_overflow=5,  # Allow N extra connections
            pool_timeout=10,  # Wait N seconds for connection
            pool_recycle=1_800,  # Recycle connections after 30 minutes
            pool_pre_ping=True,  # Test connections before use
            echo=False,
        )

        self._session_factory = sessionmaker(bind=self._engine, expire_on_commit=True)
        logger.info("Database connection pool initialized")

    @contextmanager
    def get_session(self) -> Generator[Session, None, None]:
        """Get a database session with automatic commit/rollback."""
        if not self._session_factory:
            raise RuntimeError("Session factory not initialized")

        session: Session = self._session_factory()
        try:
            yield session
            session.commit()

        except Exception:
            session.rollback()
            raise

        finally:
            session.close()

    def health_check(self) -> bool:
        """Check if database is healthy."""
        try:
            with self.get_session() as session:
                session.execute(text("SELECT 1"))
            return True
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False

    def close(self) -> None:
        """Close connection pool."""
        if self._engine:
            self._engine.dispose()
            logger.info("Database pool closed")

    @property
    def engine(self) -> Engine:
        """Get database engine."""
        if self._engine is None:
            raise RuntimeError("Database engine is not initialized.")
        return self._engine
