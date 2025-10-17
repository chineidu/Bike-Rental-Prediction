from src import create_logger
from src.db.crud import create_role, get_role_by_name
from src.db.models import Base, get_db_pool
from src.schemas.types import UserRole

logger = create_logger(name="setup")


def init_db() -> None:
    """Initialize the database by creating tables and default roles.

    This function:
    1. Creates all database tables defined in the models
    2. Creates default roles (USER, ADMIN, MODERATOR) if they don't exist

    Note
    ----
    This function should be called once when the application starts.
    It is idempotent - safe to run multiple times.
    """
    db_pool = get_db_pool()

    # Create all tables in the database (idempotent - won't recreate existing tables)
    Base.metadata.create_all(db_pool.engine)
    logger.info("Database tables created/verified")

    with db_pool.get_session() as db:
        roles_created = 0

        # Create roles if they don't exist
        if not get_role_by_name(db, UserRole.USER):
            create_role(db, UserRole.USER, "Regular user")
            roles_created += 1
            logger.info(f"Created role: {UserRole.USER}")

        if not get_role_by_name(db, UserRole.ADMIN):
            create_role(db, UserRole.ADMIN, "Administrator")
            roles_created += 1
            logger.info(f"Created role: {UserRole.ADMIN}")

        if not get_role_by_name(db, UserRole.MODERATOR):
            create_role(db, UserRole.MODERATOR, "Moderator")
            roles_created += 1
            logger.info(f"Created role: {UserRole.MODERATOR}")

    if roles_created > 0:
        logger.info(
            f"✅ Database initialized successfully! Created {roles_created} role(s)"
        )
    else:
        logger.info("✅ Database already initialized. All roles exist.")


if __name__ == "__main__":
    init_db()
