from pydantic import EmailStr, Field

from .input_schema import BaseSchema


class UserSchema(BaseSchema):
    """Schema representing a user."""

    username: str = Field(..., description="Unique identifier for the user")
    full_name: str = Field(..., description="Full name of the user")
    email: EmailStr = Field(..., description="Email address of the user")
    disabled: bool = Field(
        default=True, description="Indicates if the user is disabled"
    )


class UserInDBSchema(UserSchema):
    """Schema representing a user in the database."""

    hashed_password: str = Field(..., description="Hashed password of the user")


class UserCreateSchema(UserSchema):
    """Schema for creating a new user."""

    password: str = Field(..., description="Password for the user account")


class TokenSchema(BaseSchema):
    """Schema representing an authentication token."""

    access_token: str = Field(..., description="JWT access token")
    token_type: str = Field(..., description="Type of the token, e.g., 'bearer'")
