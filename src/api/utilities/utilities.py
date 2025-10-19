import os
import time
import warnings
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncGenerator

from fastapi import FastAPI, HTTPException, Request, status

from src import PACKAGE_PATH, create_logger
from src.api.utilities.ml_utils import PredictionService
from src.api.utilities.model_loader.loader import ModelLoader
from src.config import app_config
from src.schemas.types import ModelManagerDict
from src.utilities.utils import get_latest_model_path

warnings.filterwarnings("ignore")
logger = create_logger(name="app_utils")


MAX_WORKERS: int = os.cpu_count() - 1  # type: ignore

DUMMY_DATA: dict[str, Any] = {
    "data": [
        {
            "season": 2,
            "mnth": 6,
            "holiday": 0,
            "hr": 8,
            "weekday": 5,
            "workingday": 1,
            "weathersit": 2,
            "temp": 0.6,
            "hum": 0.83,
            "windspeed": 0.0,
            "cnt": 454,
            "datetime": "2011-06-17 08:00:00",
            "yr": 0,
        }
    ],
    "currency": "NGN",
}


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:  # noqa: ARG001
    """Initialize and cleanup FastAPI application lifecycle.

    This context manager handles the initialization of model components during startup
    and cleanup during shutdown.
    """
    try:
        start_time: float = time.perf_counter()
        logger.info("Starting up application and loading model...")

        # Load model here
        models_dir: Path = PACKAGE_PATH / "artifacts/models/"
        model_name_pattern: str = app_config.api_config.model_name_pattern
        model_extensions: list[str] = app_config.api_config.model_extensions

        # Get the most recently modified model (any type)
        for ext in model_extensions:
            model_path = get_latest_model_path(
                models_dir=models_dir,
                model_name_pattern=model_name_pattern,
                extension=ext,
            )
            if model_path:
                print(f"Loaded: {model_path.name}")
                break
        model_loader: ModelLoader = ModelLoader(model_path=model_path)  # type: ignore
        model_loader.load()
        prediction_service: PredictionService = PredictionService(model=model_loader)  # type: ignore

        # Store component in app state
        app.state.model_manager = ModelManagerDict(
            model_loader=model_loader, prediction_service=prediction_service
        )

        # Warmup model to avoid latency during first request
        start_warmup = time.perf_counter()
        try:
            _ = prediction_service.predict(
                data=DUMMY_DATA["data"],
                base_price=1000,
                base_elasticity=-1.0,
                currency=DUMMY_DATA["currency"],
            )
            warmup_time = time.perf_counter() - start_warmup
            logger.info(f"Warmup prediction completed in {warmup_time:.2f}s")

        except Exception as e:
            logger.warning(f"Warmup prediction failed: {e}")

        logger.info(
            f"Application startup completed in {time.perf_counter() - start_time:.2f} seconds"
        )

        # Yield control to the application
        yield

    except Exception as e:
        logger.error(f"Failed to load model during startup: {e}")
        raise

    finally:
        # Cleanup on shutdown
        if hasattr(app.state, "model_manager"):
            try:
                app.state.model_manager["model_loader"].unload()
                logger.info("Model unloaded during shutdown")

            except Exception as e:
                logger.error(f"Error during shutdown cleanup: {e}")


def get_model_manager(request: Request) -> ModelManagerDict:
    """Get the prediction service from the app state."""
    if (
        not hasattr(request.app.state, "model_manager")
        or request.app.state.model_manager is None
    ):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Prediction service not loaded. Please try again later.",
        )
    return request.app.state.model_manager
