from datetime import timedelta

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm

from src import create_logger
from src.api.auth import (
    authenticate_user,
    create_access_token,
    fake_users_db,
    get_current_active_user,
)
from src.config import app_settings
from src.schemas.models import UserInDBSchema, UserSchema

logger = create_logger(name="login")

router = APIRouter(tags=["login"])


@router.post("/token", status_code=status.HTTP_200_OK)
async def login_for_access_token(
    form_data: OAuth2PasswordRequestForm = Depends(),
) -> dict[str, str]:
    """
    Authenticate a user and return an OAuth2 bearer access token.
    Validates credentials supplied via an OAuth2PasswordRequestForm (dependency-injected by FastAPI).
    On successful authentication a JWT access token is created and returned in the OAuth2 bearer format.

    Parameters
    ----------
    form_data : OAuth2PasswordRequestForm
        Dependency-injected form containing 'username' and 'password' fields. Provided by FastAPI via Depends().

    Returns
    -------
    dict[str, str]
        A dictionary containing:
        - "access_token" (str): The issued JWT access token.
        - "token_type" (str): The token type; always "bearer".

    Examples
    --------
    # Example FastAPI usage (handled automatically by FastAPI):
    # POST /token with form fields 'username' and 'password'
    # Successful response:
    # {"access_token": "<jwt_token_here>", "token_type": "bearer"}

    """
    logger.info("Authenticating user...")
    user: UserInDBSchema = authenticate_user(  # type: ignore
        db=fake_users_db,
        username=form_data.username,
        password=form_data.password,
    )
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username of password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    logger.info(f"User {user.username} authenticated successfully.")
    access_token_expires: timedelta = timedelta(
        minutes=app_settings.ACCESS_TOKEN_EXPIRE_MINUTES
    )
    access_token = create_access_token(
        data={"sub": user.username},
        expires_delta=access_token_expires,
    )

    return {"access_token": access_token, "token_type": "bearer"}


@router.get("/users/me", status_code=status.HTTP_200_OK)
async def get_current_user(
    current_user: UserSchema = Depends(get_current_active_user),
) -> dict[str, str]:  # noqa: B008
    """
    Endpoint to get the current logged-in user. This endpoint is protected and requires a valid JWT token.

    Returns:
    -------
        dict
    """
    return {
        "username": current_user.username,
        "email": current_user.email,
        "full_name": current_user.full_name,
    }
