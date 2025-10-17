from typing import Any

from pydantic import EmailStr, Field

from src.schemas.input_schema import BaseSchema


class UserSchema(BaseSchema):
    """Schema representing a user."""

    username: str = Field(..., description="Unique identifier for the user")
    full_name: str = Field(..., description="Full name of the user")
    email: EmailStr = Field(..., description="Email address of the user")
    is_active: bool | None = Field(
        default=None, description="Indicates if the user is active"
    )
    roles: list[Any] | None = Field(
        default=None, description="List of roles assigned to the user"
    )


class UserInDBSchema(UserSchema):
    """Schema representing a user in the database."""

    hashed_password: str = Field(..., description="Hashed password of the user")


class UserCreateSchema(UserSchema):
    """Schema for creating a new user."""

    password: str = Field(..., description="Password for the user account")
