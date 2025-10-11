from fastapi import APIRouter, status

from src import create_logger
from src.config import app_config
from src.schemas import HealthCheck

router = APIRouter(
    prefix="/health",
    tags=["health"],
)

logger = create_logger(name="health_check")


@router.get("/", status_code=status.HTTP_200_OK)
def health_check() -> HealthCheck:
    """
    Simple health check endpoint to verify API is operational.

    Returns:
    -------
        HealthCheck: Status of the API
    """

    return HealthCheck(
        status=app_config.api_config.status, version=app_config.api_config.version
    )
