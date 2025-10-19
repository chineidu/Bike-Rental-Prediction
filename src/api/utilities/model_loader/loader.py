"""
This uses a singleton to ensure only one instance of each model loader exists.
"""

import threading
import time
from pathlib import Path
from types import TracebackType
from typing import Any, Protocol, Type

from src import create_logger
from src.utilities.utils import load_model_from_disk

logger = create_logger(name="model_loader")
type Data = list[dict[str, Any]]


# ========================================================
# ======================= PROTOCOL =======================
# ========================================================
class ModelInterface(Protocol):
    """Protocol defining the interfaces for prediction models."""

    def predict(self, data: Data) -> Any:
        """Make a prediction on the given data."""
        ...

    def predict_batch(self, data: Data) -> list[Any]:
        """Make batch predictions on the given data."""
        ...


class ModelLoader:
    """Class responsible for lazily loading and managing a machine learning model."""

    def __init__(self, model_path: Path | str) -> None:
        self.model_path: Path | str = model_path
        self._model: Any | None = None
        self._is_initialized: bool = False
        # Use Lock for thread-safe access to shared state
        self._lock: threading.Lock = threading.Lock()

    def load(self) -> None:
        """
        Load the model in a thread-safe, idempotent manner.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        if self._is_initialized:
            logger.info("Model already loaded; skipping load")
            return

        with self._lock:
            if not self._is_initialized or not self._model:
                self._load_model()
                self._is_initialized = True
                logger.debug("Model loaded successfully")

    def predict(self, data: Data) -> Any:
        """
        Make a prediction using the currently loaded model.

        Parameters
        ----------
        data : Data
            Input data for which to generate a prediction.

        Returns
        -------
        Any
            The prediction produced by the underlying model.

        Raises
        ------
        RuntimeError
            If the model has not been loaded (i.e., load() was not called or initialization
            failed).
        """
        if not self._is_initialized or not self._model:
            raise RuntimeError("Model not loaded. Call load() first.")

        model = self.get_model()

        return model.predict(data)

    def get_model(self) -> Any:
        """
        Get the currently loaded model instance.

        Returns
        -------
        Any
            The loaded model instance.

        Raises
        ------
        RuntimeError
            If the model has not been loaded (i.e., load() was not called or initialization
            failed).
        """
        if not self._is_initialized or not self._model:
            raise RuntimeError("Model not loaded. Call load() first.")

        return self._model

    def unload(self) -> None:
        """
        Unload the model and free resources in a thread-safe manner.

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
            logger.info("Model unloaded and resources freed")

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

    def _load_model(self) -> Any:
        """
        Internal method to load the model from disk.

        Parameters
        ----------
        self : object

        Returns
        -------
        None
            The method does not return a value; the loaded model is stored in
            `self._model`.
        """
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

    def __enter__(self) -> "ModelLoader":
        """
        Enter the runtime context for the model loader.

        This calls :meth:`load` to set up any required resources and
        returns the loader instance so it can be used with the `with` statement.

        Returns
        -------
        ModelLoader
            The loaded model loader instance.
        """
        self.load()
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
            self.unload()
            logger.debug(
                f"{self.__class__.__name__} context exited and resources cleaned up"
            )

        except Exception as cleanup_error:
            # Log cleanup errors
            logger.error(
                f"Error during {self.__class__.__name__} cleanup: {cleanup_error}"
            )
