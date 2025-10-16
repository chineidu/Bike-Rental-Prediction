import os
import re
from urllib.parse import quote

from dotenv import load_dotenv
from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


def fix_url_credentials(url: str) -> str:
    """
    Fix URL by properly encoding special characters in credentials.

    Parameters
    ----------
    url : str
        The URL to fix.

    Returns
    -------
    fixed_url : str
        The fixed URL.
    """
    try:
        # More flexible pattern that accepts any scheme format
        # Captures: anything://username:password@host_and_rest
        pattern = r"^([^:]+://)([^:/?#]+):([^@]+)@(.+)$"
        match = re.match(pattern, url)

        if match:
            scheme, username, password, host_part = match.groups()
            # URL encode the username and password
            # safe='' means encode all special characters
            encoded_username = quote(username, safe="")
            encoded_password = quote(password, safe="")

            # Reconstruct the URL
            fixed_url = f"{scheme}{encoded_username}:{encoded_password}@{host_part}"

            # Extract scheme name for logging
            scheme_name = scheme.rstrip("://")  # noqa: B005
            print(f"Fixed {scheme_name!r} URL encoding for special characters")

            return fixed_url

        print("WARNING: No regex match found!")
        return url

    except Exception as e:
        print(f"Could not fix URL: {e}")
        return url


class Settings(BaseSettings):
    """
    Settings class for managing application configuration.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",  # Ignore unknown env vars or to allow populating private attributes
    )

    # ======= MLFlow =======
    MLFLOW_HOST: str = "localhost"
    MLFLOW_PORT: int = 5001

    # ======= AWS/MinIO =======
    AWS_S3_HOST: str = "localhost"
    AWS_S3_PORT: int = 9000
    AWS_S3_BUCKET: str = "mlflow-artifacts"
    AWS_ACCESS_KEY_ID: str = "minioadmin"
    AWS_SECRET_ACCESS_KEY: SecretStr = SecretStr("minioadmin")
    AWS_DEFAULT_REGION: str = "us-east-1"

    # ======= DB =======
    POSTGRES_HOST: str = "localhost"
    POSTGRES_PORT: int = 5432
    POSTGRES_USER: str = "mlflow"
    POSTGRES_PASSWORD: SecretStr = SecretStr("mlflow")
    POSTGRES_DB: str = "mlflow"

    # ======= Airflow =======
    AIRFLOW_UID: int = Field(default=50000, alias="AIRFLOW_UID")
    AIRFLOW_WWW_USER_USERNAME: str = Field(
        default="airflow", alias="_AIRFLOW_WWW_USER_USERNAME"
    )
    AIRFLOW_WWW_USER_PASSWORD: SecretStr = Field(
        default=SecretStr("airflow"), alias="_AIRFLOW_WWW_USER_PASSWORD"
    )

    @field_validator("MLFLOW_PORT", "AWS_S3_PORT", "POSTGRES_PORT", mode="before")
    @classmethod
    def parse_port_fields(cls, v: str | int) -> int:
        """Parses port fields to ensure they are integers."""
        if isinstance(v, str):
            try:
                return int(v.strip())
            except ValueError:
                raise ValueError(f"Invalid port value: {v}") from None

        if isinstance(v, int) and not (1 <= v <= 65535):
            raise ValueError(f"Port must be between 1 and 65535, got {v}")

        return v

    @property
    def mlflow_tracking_uri(self) -> str:
        """
        Constructs the MLflow tracking URI.

        Returns
        -------
        str
            Complete MLflow tracking URI in the format:
            http://host:port
        """
        url: str = f"http://{self.MLFLOW_HOST}:{self.MLFLOW_PORT}"
        return fix_url_credentials(url)

    @property
    def mlflow_s3_endpoint_url(self) -> str:
        """
        Constructs the S3 endpoint URL for MLflow.

        Returns
        -------
        str
            Complete S3 endpoint URL in the format:
            http://host:port
        """
        url: str = f"http://{self.AWS_S3_HOST}:{self.AWS_S3_PORT}"
        return fix_url_credentials(url)


def refresh_settings() -> Settings:
    """Refresh environment variables and return new Settings instance.

    This function reloads environment variables from .env file and creates
    a new Settings instance with the updated values.

    Returns
    -------
    Settings
        A new Settings instance with refreshed environment variables
    """
    load_dotenv(override=True)
    return Settings()  # type: ignore


def setup_env() -> None:
    """Sets environment variables for MLflow, MinIO, and AWS."""
    os.environ["MLFLOW_TRACKING_URI"] = app_settings.mlflow_tracking_uri
    os.environ["MLFLOW_S3_ENDPOINT_URL"] = app_settings.mlflow_s3_endpoint_url
    os.environ["MLFLOW_DB_URI"] = (
        f"postgresql://{app_settings.POSTGRES_USER}:"
        f"{app_settings.POSTGRES_PASSWORD.get_secret_value()}"
        f"@{app_settings.POSTGRES_HOST}"
        f":{app_settings.POSTGRES_PORT}"
        f"/{app_settings.POSTGRES_DB}"
    )
    os.environ["MLFLOW_ARTIFACT_ROOT"] = f"s3://{app_settings.AWS_S3_BUCKET}"
    os.environ["AWS_ACCESS_KEY_ID"] = app_settings.AWS_ACCESS_KEY_ID
    os.environ["AWS_SECRET_ACCESS_KEY"] = (
        app_settings.AWS_SECRET_ACCESS_KEY.get_secret_value()
    )
    os.environ["AWS_DEFAULT_REGION"] = app_settings.AWS_DEFAULT_REGION
    os.environ["MINIO_ROOT_USER"] = app_settings.AWS_ACCESS_KEY_ID
    os.environ["MINIO_ROOT_PASSWORD"] = (
        app_settings.AWS_SECRET_ACCESS_KEY.get_secret_value()
    )


app_settings: Settings = refresh_settings()

# Call setup_env only once at startup
_setup_env_called: bool = False


def setup_env_once() -> None:
    """Sets environment variables for MLflow, MinIO, and AWS. Called only once."""
    global _setup_env_called
    if not _setup_env_called:
        setup_env()
        _setup_env_called = True


# Automatically call setup_env when the module is imported
setup_env_once()
