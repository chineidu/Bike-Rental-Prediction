"""Utility functions for loading models and artifacts from MLflow."""

from typing import Any

import mlflow

from src import create_logger
from src.exceptions import MLFlowError
from src.exp_tracking.mlflow import MLFlowTracker
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
        model_name = (
            str(model_name) if isinstance(model_name, ModelType) else model_name
        )
        tracker = MLFlowTracker(
            tracking_uri=tracking_uri, experiment_name=experiment_name
        )
        return tracker.load_model_artifact(run_id=run_id, model_name=model_name)

    except (MLFlowError, Exception) as e:
        logger.error(f"Error: {str(e)}")
        return None


def get_best_run(
    experiment_name: str,
    metric: str = "best_rmse",
    tracking_uri: str = "http://localhost:5001",
) -> dict[str, Any] | None:
    """
    Get the run ID of the best performing model based on a metric.

    Parameters
    ----------
    experiment_name : str
        Name of the MLflow experiment.
    metric : str, default="best_rmse"
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
        client = mlflow.tracking.MlflowClient()  # type: ignore

        # Get experiment
        experiment = client.get_experiment_by_name(experiment_name)
        if experiment is None:
            raise ValueError(f"Experiment '{experiment_name}' not found")

        # Search for runs sorted by metric
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=[f"metrics.{metric} ASC"],  # ASC for metrics where lower is better
            max_results=1,
        )

        if not runs:
            raise ValueError(f"No runs found in experiment '{experiment_name}'")

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
        logger.warning(f"Error: {e}")
        return None


def load_best_model(
    experiment_name: str,
    metric: str = "best_rmse",
    tracking_uri: str = "http://localhost:5001",
) -> dict[str, Any] | None:
    """
    Load the best performing model from an experiment.

    Parameters
    ----------
    experiment_name : str
        Name of the MLflow experiment.
    metric : str, default="best_rmse"
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
            metric=metric,
            tracking_uri=tracking_uri,
        )
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
        logger.warning(f"Error: {e}")
        return None
