"""Utility functions for loading models and artifacts from MLflow."""

import cloudpickle
from pathlib import Path
import json
from typing import Any

import mlflow
from mlflow import MlflowClient  # type: ignore

from src import create_logger
from src.exceptions import MLFlowError
from src.exp_tracking.mlflow_tracker import MLFlowTracker
from src.schemas.types import ModelType

logger = create_logger("model_loader")


def load_model_from_run(
    run_id: str,
    model_name: str | ModelType,
    tracking_uri: str = "http://localhost:5001",
    experiment_name: str = "bike rental",
) -> dict[str, Any] | None:
    """
    Convenience function to load a model from MLflow run.

    Parameters
    ----------
    run_id : str
        MLflow run ID containing the model.
    model_name : str | ModelType
        Name of the model (e.g., ModelType.XGBOOST, ModelType.LIGHTGBM, etc.).
    tracking_uri : str, default="http://localhost:5001"
        MLflow tracking server URI.
    experiment_name : str, default="bike rental"
        Name of the experiment.

    Returns
    -------
    dict[str, Any] | None
        Dictionary containing:
        - "model": The loaded model object
        - "metadata": Model metadata
        - "input_example": Input example data (if available)
        - "model_uri": URI to the model artifact

    """
    try:
        model_name = str(model_name)
        tracker = MLFlowTracker(
            tracking_uri=tracking_uri, experiment_name=experiment_name
        )
        return tracker.load_model_artifact(run_id=run_id, model_name=model_name)

    except (MLFlowError, Exception) as e:
        logger.error(f"❌ Error: {str(e)}")
        return None


def get_best_run(
    experiment_name: str,
    client: MlflowClient,
    metric: str = "RMSE",
    tracking_uri: str = "http://localhost:5001",
) -> dict[str, Any] | None:
    """
    Get the run ID of the best performing model based on a metric.

    Parameters
    ----------
    experiment_name : str
        Name of the MLflow experiment.
    client : MlflowClient
        An instance of MlflowClient to interact with the MLflow tracking server.
    metric : str, default="RMSE"
        Metric to use for comparison (lower is better for RMSE).
    tracking_uri : str, default="http://localhost:5001"
        MLflow tracking server URI.

    Returns
    -------
    dict[str, Any] | None
        Run ID of the best performing model.
    """
    try:
        mlflow.set_tracking_uri(tracking_uri)

        # Get experiment
        experiment = client.get_experiment_by_name(experiment_name)

        if experiment is None:
            raise ValueError(f"Experiment {experiment_name!r} not found")

        # Search for runs sorted by metric
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            # ASC for metrics where lower is better
            order_by=[f"metrics.{metric} ASC"],
            max_results=1,
        )

        if not runs:
            raise ValueError(f"No runs found in experiment {experiment_name!r}")

        best_run = runs[0]
        best_run_id = best_run.info.run_id
        best_metric_value = best_run.data.metrics.get(metric)

        logger.info(f"Best run: {best_run_id} with {metric}={best_metric_value}")
        return {
            "run_id": best_run_id,
            "metric_value": best_metric_value,
            "data": best_run,
        }

    except (MLFlowError, Exception) as e:
        logger.warning(f"❌ Error: {e}")
        return None


def load_best_model(
    experiment_name: str,
    client: MlflowClient,
    metric: str = "RMSE",
    tracking_uri: str = "http://localhost:5001",
) -> dict[str, Any] | None:
    """
    Load the best performing model from an experiment.

    Parameters
    ----------
    experiment_name : str
        Name of the MLflow experiment.
    client : MlflowClient
        An instance of MlflowClient to interact with the MLflow tracking server.
    metric : str, default="RMSE"
        Metric to use for comparison (lower is better for RMSE).
    tracking_uri : str, default="http://localhost:5001"
        MLflow tracking server URI.

    Returns
    -------
    dict[str, Any] | None
        Dictionary containing the best model and its metadata.

    """
    try:
        best_run: dict[str, Any] | None = get_best_run(
            experiment_name=experiment_name,
            client=client,
            metric=metric,
            tracking_uri=tracking_uri,
        )
        print(f"Best Run: {best_run}")
        model_name = best_run["data"].data.tags["model_family"] if best_run else None

        if best_run is None or model_name is None:
            logger.warning("No best run found or model name is missing")
            return None

        result = load_model_from_run(
            run_id=best_run["run_id"],  # type: ignore
            model_name=model_name,
            tracking_uri=tracking_uri,
            experiment_name=experiment_name,
        )
        if result is None:
            logger.warning("Failed to load model from the best run")
            return None

        logger.info(f"✅ Loaded best {model_name} model")
        return result

    except (MLFlowError, Exception) as e:
        logger.warning(f"❌Error: {e}", exc_info=True)
        return None


def set_registered_model_alias(
    client: MlflowClient,
    model_name: str,
    model_version_alias: str,
    version: str,
    environment: str = "staging",
) -> dict[str, Any] | None:
    """Set the alias for a registered model version for easy access.

    Parameters
    ----------
    client : MlflowClient
        An instance of MlflowClient to interact with the MLflow tracking server.
    model_name : str
        The name of the registered model.
    model_version_alias : str
        The alias to set for the model version (e.g., "staging", "production").
    version : str
        The version number of the model to which the alias should point.
    environment : str, default="staging"
        The environment for which the alias is being set.

    Returns
    -------
    dict[str, Any] | None
    """
    try:
        # Set model version alias
        model_name_fmtd: str = f"{model_name}_{environment}"
        value: dict[str, Any] = {
            "model_name": model_name_fmtd,
            "version": version,
            "environment": environment,
        }
        # Duplicate of step in UI
        client.set_registered_model_alias(model_name_fmtd, model_version_alias, version)
        client.set_model_version_tag(
            model_name_fmtd, version, key="extras", value=json.dumps(value)
        )
        logger.info(
            f"✅ Set alias {model_version_alias!r} for model {model_name_fmtd!r} version {version!r}"
        )

        return {
            "status": "success",
            "model_name": model_name_fmtd,
            "version": version,
            "model_version_alias": model_version_alias,
        }

    except Exception as e:
        logger.error(f"❌ Error setting model alias: {e}")
        return None


def load_registered_model_from_registry(
    model_name: str,
    model_version_alias: str = "staging",
    artifact_path: str = "models",
    tracking_uri: str = "http://localhost:5001",
) -> Any:
    """
    Load a registered MLflow model from the model registry.

    The supported model flavors are: sklearn, xgboost, and lightgbm.

    Parameters
    ----------
    model_name : str
        Name of the registered model in the MLflow Registry.
    model_version_alias : str, optional
        Version alias or stage to load (for example "staging", "production",
        or a numeric version string), by default "staging".
    artifact_path : str, optional
        Artifact subpath to include when building the model URI. The function
        builds the URI as "{artifact_path}:/{model_name}@{model_version_alias}",
        by default "models".
    tracking_uri : str, optional
        MLflow tracking server URI, by default "http://localhost:5001".

    Returns
    -------
    Any or None
        The loaded model object if loading succeeded using one of the available
        MLflow flavor loaders. If no loader is available or all loaders fail,
        None is returned.
    """
    import os

    # Set the MLflow tracking URI
    mlflow.set_tracking_uri(tracking_uri)

    # Set MinIO/S3 endpoint and credentials
    s3_endpoint = os.getenv("MLFLOW_S3_ENDPOINT_URL")
    aws_access_key = os.getenv("AWS_ACCESS_KEY_ID")
    aws_secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")

    if s3_endpoint:
        os.environ["MLFLOW_S3_ENDPOINT_URL"] = s3_endpoint
        logger.info(f"Using S3 endpoint: {s3_endpoint}")

    if aws_access_key and aws_secret_key:
        os.environ["AWS_ACCESS_KEY_ID"] = aws_access_key
        os.environ["AWS_SECRET_ACCESS_KEY"] = aws_secret_key
        logger.info("AWS credentials configured for S3 access")
    else:
        logger.warning("⚠️ AWS credentials not found in environment")

    # Get the model version using a model URI
    model_uri: str = f"{artifact_path}:/{model_name}@{model_version_alias}"
    model: Any | None = None

    loaders: list[tuple[str, Any]] = [
        ("sklearn", getattr(mlflow, "sklearn", None)),
        ("xgboost", getattr(mlflow, "xgboost", None)),
        ("lightgbm", getattr(mlflow, "lightgbm", None)),
    ]

    for name, module in loaders:
        if module is None:
            logger.warning(f"⚠️ MLflow does not expose loader for {name!r}; skipping.")
            continue

        loader = getattr(module, "load_model", None)
        if loader is None:
            logger.warning(f"⚠️ Loader for {name}!r does not have load_model; skipping.")
            continue

        # Load the model
        try:
            model = loader(model_uri)
            logger.info(f"✅ Successfully loaded model {model_name} using {name!r}")
            break

        except Exception as e:
            logger.warning(f"❌ Failed to load with {name!r}: {e}")
            continue

    if model is None:
        logger.warning(
            "❌ Failed to load model after trying sklearn, xgboost and lightgbm."
        )
        return None

    return model


def download_model(
    model: Any, model_name: str, model_version: str, models_dir: Path
) -> Path:
    """
    Save a trained model object to disk using an appropriate serialization method
    based on the model's library and type.

    Parameters
    ----------
    model : Any
        The trained model object to be saved. Supported model types include:
        - scikit-learn estimators (detected by "sklearn" in the model's module) —
          serialized with cloudpickle to a .pkl file.
        - XGBoost models (detected by "xgboost" in the module or class names
          "Booster", "XGBClassifier", "XGBRegressor") — saved via model.save_model()
          to a .json file.
        - LightGBM models (detected by "lightgbm" in the module or class names
          "Booster", "LGBMClassifier", "LGBMRegressor") — saved via model.save_model()
          to a .txt file.
    model_name : str
        Base filename to use when saving the model (the function will append an
        extension and the version string).
    model_version : str
        Version identifier to include in the filename (e.g. "v1", "2023-01-01").
    models_dir : pathlib.Path or str
        Directory where the model file will be written. If a string is provided it
        will be converted to a pathlib.Path.
    Returns
    -------
    Path
        The saved model's path.
    """
    if isinstance(models_dir, str):
        models_dir = Path(models_dir)

    model_type_name: str = type(model).__name__
    model_module: str = type(model).__module__

    # Check if it's an sklearn model (from sklearn.* modules)
    if "sklearn" in model_module:
        model_filename: str = f"{model_name}_{model_version}.pkl"
        model_path: Path = models_dir / model_filename
        with open(model_path, "wb") as fh:
            cloudpickle.dump(model, fh)
        logger.info("💾 Saved sklearn model using cloudpickle")

    # Check if it's an XGBoost model
    elif "xgboost" in model_module or model_type_name in [
        "Booster",
        "XGBClassifier",
        "XGBRegressor",
    ]:
        model_filename = f"{model_name}_{model_version}.json"
        model_path = models_dir / model_filename
        model.save_model(str(model_path))
        logger.info("💾 Saved XGBoost model using .save_model()")

    # Check if it's a LightGBM model
    elif "lightgbm" in model_module or model_type_name in [
        "Booster",
        "LGBMClassifier",
        "LGBMRegressor",
    ]:
        model_filename = f"{model_name}_{model_version}.txt"
        model_path = models_dir / model_filename
        model.save_model(str(model_path))
        logger.info("💾 Saved LightGBM model using .save_model()")

    return model_path
