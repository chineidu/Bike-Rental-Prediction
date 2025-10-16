import asyncio
import os
import threading
import time
import warnings
from contextlib import asynccontextmanager
from pathlib import Path
from types import TracebackType
from typing import Any, AsyncGenerator, Type

import httpx
from fastapi import FastAPI, HTTPException, Request, status

from src import PACKAGE_PATH, create_logger
from src.config import app_config
from src.utilities.utils import get_latest_model_path, load_model_from_disk

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


class ModelManager:
    """
    Manage lazy, thread-safe loading and lifecycle of a machine learning model stored on disk.
    """

    def __init__(self, model_path: Path | str) -> None:
        self.model_path: Path | str = model_path
        self._model: Any | None = None
        self._is_initialized: bool = False
        # Use RLock for reentrant locking. i.e. the same thread can acquire the lock multiple times.
        self._lock: threading.RLock = threading.RLock()

    def initialize(self) -> None:
        """
        Initialize the model in a thread-safe, idempotent manner.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        with self._lock:
            if self._is_initialized:
                logger.info("Model already initialized")
                return
            self._load_model()
            self._is_initialized = True
            logger.debug("Model initialized successfully")

    def _load_model(self) -> Any:
        """
        Load the machine learning model from disk and update internal state.

        Parameters
        ----------
        self : object

        Returns
        -------
        None
            The method does not return a value; the loaded model is stored in
            `self._model`.
        """
        with self._lock:
            try:
                start_time: float = time.perf_counter()
                self._model = load_model_from_disk(model_path=self.model_path)
                load_time: float = time.perf_counter() - start_time
                logger.debug(f"Model successfully loaded in {load_time:.2f} seconds")

            except Exception as e:
                logger.error(f"Error loading model: {e}")
                self._model = None
                self._is_initialized = False
                raise Exception("Error loading model and dependencies") from e

    def get_model(self) -> Any:
        """
        Retrieve the initialized model instance.

        Acquires an internal lock and returns the model previously set by the
        initialize() method. This ensures thread-safe access to the shared model
        resource.

        Returns
        -------
        Any
            The initialized model object.
        """
        with self._lock:
            if not self._is_initialized or not self._model:
                raise RuntimeError("Model is not initialized. Call initialize() first")
            return self._model

    def clear_cache(self) -> None:
        """
        Clear the cached model and reset initialization state.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        with self._lock:
            if self._is_initialized or self._model:
                del self._model
            self._model = None
            self._is_initialized = False
            logger.info("Cache cleared")

    def is_ready(self) -> bool:
        """
        Return whether the instance is ready for use.

        Returns
        -------
        bool
            True if the instance is initialized and a model is loaded, False otherwise.
        """
        with self._lock:
            return self._is_initialized and self._model is not None

    def __enter__(self) -> "ModelManager":
        """
        Enter the runtime context for the model manager.

        This calls :meth:`initialize` to set up any required resources and
        returns the manager instance so it can be used with the `with` statement.

        Returns
        -------
        ModelManager
            The initialized model manager instance.
        """
        self.initialize()
        return self

    def __exit__(
        self,
        exc_type: Type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Context manager exit - cleanup resources.

        Called when exiting a with-statement.

        Parameters
        ----------
        exc_type : Type[BaseException]
            The exception class raised in the with-block, or None if no exception was raised.
        exc_val : BaseException
            The exception instance raised in the with-block, or None if no exception was raised.
        exc_tb : TracebackType | None
            The traceback object associated with the exception, or None if no exception was raised.

        Returns
        -------
        None
            Returning None indicates that any exception raised inside the with-block
            should be propagated (i.e., this method does not suppress exceptions).
        """
        try:
            # Log if an exception occurred in the with-block
            if exc_type is not None:
                logger.warning(
                    f"Exception occurred in {self.__class__.__name__} context: {exc_type.__name__}: {exc_val}"
                )

            # Always perform cleanup, even if an exception occurred
            self.clear_cache()
            logger.debug(
                f"{self.__class__.__name__} context exited and resources cleaned up"
            )

        except Exception as cleanup_error:
            # Log cleanup errors but don't suppress original exception
            logger.error(
                f"Error during {self.__class__.__name__} cleanup: {cleanup_error}"
            )


class ModelManagerFactory:
    """Factory for creating and caching ModelManager instances per process."""

    _instances: dict[int, ModelManager] = {}
    _lock: threading.RLock = threading.RLock()  # Use RLock for consistency

    @classmethod
    def get_manager(cls, model_path: str) -> ModelManager:
        """Get or create a ModelManager instance for the current process.

        Parameters
        ----------
        model_path : str
            Path to the model weights
        memory_fraction : float, optional
            GPU memory fraction, by default 0.8

        Returns
        -------
        ModelManager
            A ModelManager instance for this process
        """
        import os

        pid: int = os.getpid()

        with cls._lock:
            if pid not in cls._instances:
                logger.info(f"Creating ModelManager for process {pid}")
                manager = ModelManager(model_path)
                manager.initialize()
                cls._instances[pid] = manager

            return cls._instances[pid]

    @classmethod
    def clear_all(cls) -> None:
        """Clear all cached instances."""
        with cls._lock:
            for manager in cls._instances.values():
                manager.clear_cache()
            cls._instances.clear()


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
        model_manager: ModelManager = ModelManager(model_path=model_path)  # type: ignore
        model_manager.initialize()

        # Store component in app state
        app.state.model_manager = model_manager

        # Wait a few seconds before making the first request
        # await asyncio.sleep(1)
        # # Warmup model to avoid latency during first request
        # asyncio.create_task(perform_http_warmup())
        logger.info(
            f"Model initialized in {time.perf_counter() - start_time:.2f} seconds"
        )

        # Yield control to the application
        yield

    except Exception as e:
        logger.error(f"Failed to initialize model during startup: {e}")
        raise

    finally:
        # Cleanup on shutdown
        if hasattr(app.state, "model_manager"):
            try:
                app.state.model_manager.clear_cache()
                logger.info("Model cache cleared during shutdown")

            except Exception as e:
                logger.error(f"Error during shutdown cleanup: {e}")


def get_model_manager(request: Request) -> dict[str, Any]:
    """Get the model components from the app state."""
    if (
        not hasattr(request.app.state, "model_manager")
        or request.app.state.model_manager is None
    ):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model manager not loaded. Please try again later.",
        )
    return request.app.state.model_manager.get_model()


async def perform_http_warmup() -> None:
    """
    Perform HTTP warmup request after the server has started.

    This function sends a POST request to the prediction endpoint with dummy data
    to prevent cold start delays.
    """
    # Wait for server to be fully initialized
    await asyncio.sleep(1)

    logger.info("Performing HTTP warmup request...")
    logger.info(f"Port: {app_config.api_config.server.port}")

    url: str = f"http://127.0.0.1:{app_config.api_config.server.port}{app_config.api_config.prefix}/predict"
    try:
        async with httpx.AsyncClient() as client:
            response: httpx.Response = await client.post(
                url,
                # json=DUMMY_DATA,
                timeout=5.0,
            )
            response.raise_for_status()
            if response.status_code in (200, 202):
                logger.info("HTTP warmup request successful")
            else:
                logger.warning(
                    f"HTTP warmup request failed with status: {response.status_code}"
                )
    except Exception as e:
        logger.error(f"Error during HTTP warmup request: {e}")
