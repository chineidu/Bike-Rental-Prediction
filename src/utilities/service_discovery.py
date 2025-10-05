import os

import httpx

from src import create_logger
from src.config import app_settings

logger = create_logger("service_discovery")
MLFLOW_PORT = app_settings.MLFLOW_PORT
MINIO_PORT = app_settings.AWS_S3_PORT


def get_mlflow_endpoint() -> str:
    """
    Find MLflow endpoint, trying environment variable first, then common endpoints

    Parameters
    ----------
    None

    Returns
    -------
    str
        MLflow endpoint as a string
    """
    # Check environment variable first
    if env_uri := os.getenv("MLFLOW_TRACKING_URI"):
        return env_uri

    # Check if we're in a container
    in_container = (
        os.path.exists("/.dockerenv") or "AIRFLOW__CORE__EXECUTOR" in os.environ
    )

    # Define endpoints based on environment
    endpoints = (
        [
            f"http://mlflow:{MLFLOW_PORT}",
            f"http://host.docker.internal:{MLFLOW_PORT}",
            f"http://172.17.0.1:{MLFLOW_PORT}",
            f"http://localhost:{MLFLOW_PORT}",
        ]
        if in_container
        else [
            f"http://localhost:{MLFLOW_PORT}",
            f"http://127.0.0.1:{MLFLOW_PORT}",
            f"http://host.docker.internal:{MLFLOW_PORT}",
        ]
    )

    # Test each endpoint
    for endpoint in endpoints:
        try:
            response = httpx.get(endpoint, timeout=2.0)
            response.raise_for_status()
            logger.info(f"MLflow accessible at: {endpoint}")
            return endpoint

        except Exception as e:
            logger.debug(f"MLflow not accessible at {endpoint}: {e}")

    # Return default if none work
    default = (
        f"http://mlflow:{MLFLOW_PORT}"
        if in_container
        else f"http://localhost:{MLFLOW_PORT}"
    )
    logger.warning(f"Could not connect to MLflow, using default: {default}")
    return default


def get_minio_endpoint() -> str:
    """
    Find MinIO endpoint, trying environment variable first, then common endpoints

    Parameters
    ----------
    None

    Returns
    -------
    str
        MinIO endpoint as a string
    """
    # Check environment variable first
    if env_uri := os.getenv("MLFLOW_S3_ENDPOINT_URL"):
        return env_uri

    # Check if we're in a container
    in_container = (
        os.path.exists("/.dockerenv") or "AIRFLOW__CORE__EXECUTOR" in os.environ
    )

    # Define endpoints based on environment
    endpoints = (
        [
            f"http://minio:{MINIO_PORT}",
            f"http://host.docker.internal:{MINIO_PORT}",
            f"http://172.17.0.1:{MINIO_PORT}",
            f"http://localhost:{MINIO_PORT}",
        ]
        if in_container
        else [
            f"http://localhost:{MINIO_PORT}",
            f"http://127.0.0.1:{MINIO_PORT}",
            f"http://host.docker.internal:{MINIO_PORT}",
        ]
    )

    # Test each endpoint
    for endpoint in endpoints:
        try:
            response = httpx.get(endpoint, timeout=2.0)
            response.raise_for_status()
            logger.info(f"MinIO accessible at: {endpoint}")
            return endpoint

        except Exception as e:
            logger.debug(f"MinIO not accessible at {endpoint}: {e}")

    # Return default if none work
    default = (
        f"http://minio:{MINIO_PORT}"
        if in_container
        else f"http://localhost:{MINIO_PORT}"
    )
    logger.warning(f"Could not connect to MinIO, using default: {default}")
    return default
