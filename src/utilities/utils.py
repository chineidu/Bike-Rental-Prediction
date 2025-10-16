import time
from functools import wraps
from pathlib import Path
from typing import Any, Callable

import joblib

from src import create_logger

logger = create_logger(name="utils")


def get_vibrant_color(correlation: float) -> str:
    """Generate vibrant colors based on correlation strength and direction."""
    abs_corr = abs(correlation)

    if correlation > 0:
        # Positive correlations: vibrant blues to greens
        if abs_corr > 0.7:
            return "#00FF41"  # Bright green
        if abs_corr > 0.5:
            return "#00D4AA"  # Teal
        if abs_corr > 0.3:
            return "#00B4D8"  # Cyan
        return "#0077BE"  # Blue
    # Negative correlations: vibrant reds to oranges
    if abs_corr > 0.7:
        return "#FF073A"  # Bright red
    if abs_corr > 0.5:
        return "#FF4500"  # Orange red
    if abs_corr > 0.3:
        return "#FF6B35"  # Orange
    return "#FFB000"  # Amber


def _select_valid_columns(
    actual_cols: list[str], selected_cols: list[str]
) -> list[str]:
    """Select valid columns from the actual columns based on user selection."""
    return list(set(actual_cols) & set(selected_cols))


def load_model_from_disk(
    model_path: str | Path,
) -> Any:
    """
    Load a trained model from disk.

    This function loads models that were saved using the download_model function.
    It supports:
    - scikit-learn models (.pkl files) - loaded with joblib
    - XGBoost models (.json files) - loaded with xgboost
    - LightGBM models (.txt files) - loaded with lightgbm

    Parameters
    ----------
    model_path : str or Path
        Path to the model file to load.

    Returns
    -------
    Any
        The loaded model object.

    Examples
    --------
    >>> # Load an sklearn model
    >>> model = load_model_from_disk("/path/to/model.pkl")
    >>> predictions = model.predict(X_test)
    """
    model_path = Path(model_path)

    if not model_path.exists():
        raise FileNotFoundError(f"❌ Model file not found: {model_path}")

    # Load based on file extension
    if model_path.suffix == ".pkl":
        # sklearn model saved with joblib
        return joblib.load(model_path)

    if model_path.suffix == ".json":
        # XGBoost model
        import xgboost as xgb

        model = xgb.Booster()
        model.load_model(str(model_path))
        return model

    if model_path.suffix == ".txt":
        # LightGBM model
        import lightgbm as lgb

        return lgb.Booster(model_file=str(model_path))

    raise ValueError(
        f"❌ Unsupported model file extension: {model_path.suffix}. "
        "Supported: .pkl (sklearn/joblib), .json (xgboost), .txt (lightgbm)"
    )


def load_all_models_from_directory(
    models_dir: str | Path,
    extensions: list[str] | None = None,
) -> dict[str, Any]:
    """
    Load all model files from a directory.

    Parameters
    ----------
    models_dir : str or Path
        Directory containing model files.
    extensions : list[str], optional
        List of file extensions to load (e.g., [".pkl", ".json", ".txt"]).
        If None, defaults to [".pkl", ".json", ".txt"].

    Returns
    -------
    dict[str, Any]
        Dictionary mapping model filenames (without path) to loaded model objects.

    Examples
    --------
    >>> # Load all models from artifacts directory
    >>> models = load_all_models_from_directory("/opt/airflow/artifacts/models")
    >>> print(models.keys())
    dict_keys(['ModelType.RANDOM_FOREST_staging_1.pkl', ...])
    >>>
    >>> # Use a specific model
    >>> rf_model = models['ModelType.RANDOM_FOREST_staging_1.pkl']
    >>> predictions = rf_model.predict(X_test)
    """
    models_dir = Path(models_dir)

    if not models_dir.exists():
        raise FileNotFoundError(f"Models directory not found: {models_dir}")

    if extensions is None:
        extensions = [".pkl", ".json", ".txt"]

    models: dict[str, Any] = {}

    # Iterate through all files in directory
    for model_file in models_dir.iterdir():
        if model_file.is_file() and model_file.suffix in extensions:
            try:
                model = load_model_from_disk(model_file)
                models[model_file.name] = model
                print(f"✅ Loaded: {model_file.name}")
            except Exception as e:
                print(f"❌ Failed to load {model_file.name}: {e}")

    return models


def get_latest_model_path(
    models_dir: str | Path,
    model_name_pattern: str = "*",
    extension: str = ".pkl",
) -> Path | None:
    """
    Get the path to the most recently modified model file matching a pattern.

    Parameters
    ----------
    models_dir : str or Path
        Directory containing model files.
    model_name_pattern : str, optional
        Glob pattern to match model filenames (default: "*" matches all).
        Examples: "RandomForest*", "*staging*", "ModelType.XGBOOST*"
    extension : str, optional
        File extension to filter by (default: ".pkl").

    Returns
    -------
    Path or None
        Path to the most recent model file, or None if no matches found.

    Examples
    --------
    >>> # Get latest RandomForest model
    >>> model_path = get_latest_model_path(
    ...     "/opt/airflow/artifacts/models",
    ...     model_name_pattern="*RANDOM_FOREST*"
    ... )
    >>> model = load_model_from_disk(model_path)
    """
    models_dir = Path(models_dir)

    if not models_dir.exists():
        raise FileNotFoundError(f"Models directory not found: {models_dir}")

    # Find all matching files
    pattern = f"{model_name_pattern}{extension}"
    matching_files = list(models_dir.glob(pattern))

    if not matching_files:
        return None

    # Return the most recently modified file
    return max(matching_files, key=lambda p: p.stat().st_mtime)


def log_function_duration(func: Callable[..., Any]) -> Callable[..., Any]:
    """
    Decorator to log the duration of a function execution.

    Parameters
    ----------
    func : Callable[..., Any]
        The function to be decorated.

    Returns
    -------
    Callable[..., Any]
        The wrapped function with duration logging.
    """

    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        start_time = time.perf_counter()

        # Call the original function
        result = func(*args, **kwargs)

        print(
            f"Function {func.__name__!r} executed in {time.perf_counter() - start_time:.4f} seconds"  # type: ignore
        )  # type: ignore

        return result

    return wrapper


def async_log_function_duration(func: Callable[..., Any]) -> Callable[..., Any]:
    """
    Decorator to log the duration of an asynchronous function execution.

    Parameters
    ----------
    func : Callable[..., Any]
        The asynchronous function to be decorated.

    Returns
    -------
    Callable[..., Any]
        The wrapped asynchronous function with duration logging.
    """

    @wraps(func)
    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        start_time = time.perf_counter()
        func_name = getattr(func, "__name__", "unknown")

        try:
            logger.debug(f"Starting async function {func_name!r}")

            # Await the original async function
            result = await func(*args, **kwargs)

            elapsed_time = time.perf_counter() - start_time
            logger.info(
                f"Async function {func_name!r} completed in {elapsed_time:.4f} seconds"
            )

            return result

        except Exception as e:
            elapsed_time = time.perf_counter() - start_time
            logger.error(
                f"Async function {func_name!r} failed after {elapsed_time:.4f} seconds: {e}"
            )
            raise

    return wrapper
