import asyncio
import os
import time
import warnings
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator

import httpx
from fastapi import FastAPI

from src import create_logger
from src.config import app_config

warnings.filterwarnings("ignore")
logger = create_logger(name="app_utils")

MAX_WORKERS: int = os.cpu_count() - 1  # type: ignore

DUMMY_DATA: dict[str, Any] = {
    "data": [
        {
            "id": "1",
            "text": "treehouse cart payment communion retail store with levy through merrybet",
        },
        {"id": "2", "text": "alat transfer opay get drugs"},
    ]
}


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:  # noqa: ARG001
    """Initialize and cleanup FastAPI application lifecycle.

    This context manager handles the initialization of model components during startup
    and cleanup during shutdown.
    """
    try:
        start_time: float = time.perf_counter()
        logger.info("Loading model during application startup...")

        # Load model here
        pass
        logger.info(
            f"Model initialized in {time.perf_counter() - start_time:.2f} seconds"
        )

    except Exception as e:
        logger.error(f"Failed to initialize model during startup: {e}")
        raise

    # Separates startup from shutdown
    yield
    logger.info("Application shutdown...")


async def perform_http_warmup() -> None:
    """
    Perform HTTP warmup request after the server has started.

    This function sends a POST request to the prediction endpoint with dummy data
    to prevent cold start delays.
    """
    # Wait for server to be fully initialized
    await asyncio.sleep(1)

    logger.info("Performing HTTP warmup request...")
    print(f"Port: {app_config.api_config.server.port}")
    # TODO: Update endpoint as needed
    url: str = f"http://127.0.0.1:{app_config.api_config.server.port}{app_config.api_config.prefix}/extract-labels"
    try:
        async with httpx.AsyncClient() as client:
            response: httpx.Response = await client.post(
                url, json=DUMMY_DATA, timeout=30.0
            )
            if response.status_code == 200 or response.status_code == 202:
                logger.info("HTTP warmup request successful")
            else:
                logger.warning(
                    f"HTTP warmup request failed with status: {response.status_code}"
                )
    except Exception as e:
        logger.error(f"Error during HTTP warmup request: {e}")
