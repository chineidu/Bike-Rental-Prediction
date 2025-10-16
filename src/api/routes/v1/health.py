from fastapi import APIRouter, Depends, status

from src import create_logger
from src.api.utilities.utilities import ModelManager, get_model_manager
from src.config import app_config
from src.schemas import HealthCheck

router = APIRouter(tags=["health"])

logger = create_logger(name="health_check")


@router.get("/health", status_code=status.HTTP_200_OK)
def health_check(
    model_manager: ModelManager = Depends(get_model_manager),
) -> HealthCheck:  # noqa: B008
    """
    Simple health check endpoint to verify API is operational.

    Returns:
    -------
        HealthCheck: Status of the API
    """
    logger.info("Health check endpoint called.")
    is_model_ready: bool = model_manager.is_ready()

    return HealthCheck(
        status=app_config.api_config.status,
        api_name=app_config.api_config.name,
        version=app_config.api_config.version,
        model_status=is_model_ready,
    )
