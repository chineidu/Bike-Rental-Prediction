from fastapi.security import OAuth2PasswordBearer
from passlib.context import CryptContext
from sqlalchemy.orm import Session

from src.config import app_config
from src.db.models import DBRole, DBUser
from src.schemas.models import UserCreateSchema, UserSchema
from src.schemas.types import UserRole

prefix: str = app_config.api_config.prefix

# =========== Password hashing context ===========
# Using `scrypt` instead of `bcrypt` to avoid compatibility issues on macOS
pwd_context = CryptContext(schemes=["scrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl=f"{prefix}/token")


def get_password_hash(password: str) -> str:
    """Hash a password."""
    return pwd_context.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against its hash."""
    return pwd_context.verify(plain_password, hashed_password)


def get_user_by_id(db: Session, user_id: int) -> DBUser | None:
    """Get a user by their ID."""
    return db.query(DBUser).filter(DBUser.id == user_id).first()


def get_user_by_username(db: Session, username: str) -> DBUser | None:
    """Get a user by their username."""
    return db.query(DBUser).filter(DBUser.username == username).first()


def get_user_by_email(db: Session, email: str) -> DBUser | None:
    """Get a user by their email."""
    return db.query(DBUser).filter(DBUser.email == email).first()


def get_role_by_name(db: Session, name: UserRole) -> DBRole | None:
    """Get role by name."""
    return db.query(DBRole).filter(DBRole.name == name).first()


def create_user(db: Session, user: UserCreateSchema) -> DBUser:
    """Create a new user in the database."""
    # Hash the password
    hashed_password: str = get_password_hash(user.password)

    # Create user
    db_user = DBUser(
        username=user.username,
        full_name=user.full_name,
        email=user.email,
        hashed_password=hashed_password,
        is_active=user.is_active,
    )

    # Assign default role "user" to new user
    default_role = db.query(DBRole).filter(DBRole.name == UserRole.USER).first()
    if default_role:
        # Assign role
        db_user.roles.append(default_role)

    db.add(db_user)
    db.commit()
    db.refresh(db_user)

    return db_user


def authenticate_user(db: Session, username: str, password: str) -> DBUser | None:
    """Authenticate user with username and password."""
    user = get_user_by_username(db, username)
    if not user:
        return None

    if not verify_password(password, user.hashed_password):
        return None

    return user


def create_role(db: Session, role_name: UserRole, description: str) -> DBRole:
    """Create a new role in the database."""
    db_role = DBRole(name=role_name, description=description)

    db.add(db_role)
    db.commit()
    db.refresh(db_role)

    return db_role


def get_all_db_users(db: Session, skip: int, limit: int) -> list[DBUser]:
    """Get all users from the database with pagination."""
    return db.query(DBUser).offset(skip).limit(limit).all()


def convert_db_user_to_user_schema(db_user: DBUser) -> UserSchema:
    """Convert a DBUser instance to a UserSchema instance."""
    return UserSchema(
        username=db_user.username,
        full_name=db_user.full_name,
        email=db_user.email,
        is_active=db_user.is_active,
        roles=[role.name for role in db_user.roles],
    )
