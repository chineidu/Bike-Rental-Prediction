from datetime import datetime, timedelta
from typing import Any

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from passlib.context import CryptContext

from src import create_logger
from src.config import app_config, app_settings
from src.schemas.models import UserInDBSchema, UserSchema

logger = create_logger(name="auth")

# =========== Configuration ===========
SECRET_KEY: str = app_settings.SECRET_KEY.get_secret_value()
ALGORITHM: str = app_settings.ALGORITHM
ACCESS_TOKEN_EXPIRE_MINUTES: int = app_settings.ACCESS_TOKEN_EXPIRE_MINUTES

prefix: str = app_config.api_config.prefix

# =========== Password hashing context ===========
# Using `scrypt` instead of `bcrypt` to avoid compatibility issues on macOS
pwd_context = CryptContext(schemes=["scrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl=f"{prefix}/token")


# TODO: Replace with real user database
fake_users_db: dict[str, UserInDBSchema] = {
    "johndoe": UserInDBSchema(
        username="johndoe",
        full_name="John Doe",
        email="johndoe@example.com",
        hashed_password=pwd_context.hash("password"),
        disabled=False,
    )
}


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a plain password against its hashed version."""
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    """Hash a password."""
    return pwd_context.hash(password)


# TODO: Replace with real user retrieval logic
def get_user(db: dict[str, UserInDBSchema], username: str) -> UserInDBSchema | None:
    """Retrieve a user from the database."""
    return db.get(username)  # type: ignore


def authenticate_user(
    db: dict[str, UserInDBSchema], username: str, password: str
) -> UserInDBSchema | None:
    """Authenticate a user."""
    user = get_user(db, username)
    if not user or not verify_password(password, user.hashed_password):
        return None
    return user


# =========== JWT Token Management ===========
def create_access_token(
    data: dict[str, str], expires_delta: timedelta | None = None
) -> str:
    """Create a JWT access token."""
    to_encode: dict[str, Any] = data.copy()
    expire: datetime = datetime.now() + (expires_delta or timedelta(minutes=15))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


async def get_current_user(token: str = Depends(oauth2_scheme)) -> UserSchema:
    """Get the current user from the JWT token."""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload: dict[str, Any] = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str | None = payload.get("sub")

        if username is None:
            raise credentials_exception

    except JWTError as e:
        raise credentials_exception from e

    user = get_user(fake_users_db, username)
    if user is None:
        raise credentials_exception
    return UserSchema(**user.model_dump())  # type: ignore


async def get_current_active_user(
    current_user: UserSchema = Depends(get_current_user),
) -> UserSchema:  # noqa: B008
    """Get the current active user."""
    if current_user.disabled:  # type: ignore
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user
