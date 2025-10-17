from datetime import datetime, timedelta
from typing import Any

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from sqlalchemy.orm import Session

from src import create_logger
from src.config import app_config, app_settings
from src.db.crud import convert_db_user_to_user_schema, get_user_by_username
from src.db.models import get_db
from src.schemas.models import UserSchema

logger = create_logger(name="auth")
prefix: str = app_config.api_config.prefix
# =========== Configuration ===========
SECRET_KEY: str = app_settings.SECRET_KEY.get_secret_value()
ALGORITHM: str = app_settings.ALGORITHM
ACCESS_TOKEN_EXPIRE_MINUTES: int = app_settings.ACCESS_TOKEN_EXPIRE_MINUTES
oauth2_scheme = OAuth2PasswordBearer(tokenUrl=f"{prefix}/token")


# =========== JWT Token Management ===========
def create_access_token(
    data: dict[str, str], expires_delta: timedelta | None = None
) -> str:
    """Create a JWT access token."""
    to_encode: dict[str, Any] = data.copy()
    expire: datetime = datetime.now() + (expires_delta or timedelta(minutes=15))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


async def get_current_user(
    token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)
) -> UserSchema:
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

    db_user = get_user_by_username(db=db, username=username)
    if db_user is None:
        raise credentials_exception
    return convert_db_user_to_user_schema(db_user)


async def get_current_active_user(
    current_user: UserSchema = Depends(get_current_user),
) -> UserSchema:  # noqa: B008
    """Get the current active user."""
    if not current_user.is_active:  # type: ignore
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user
