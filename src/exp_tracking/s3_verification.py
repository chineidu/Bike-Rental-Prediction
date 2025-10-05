"""This module provides utilities for S3 artifact verification.
Inspired by: https://github.com/airscholar/astro-salesforecast/blob/main/include/utils/s3_verification.py
"""

from typing import Any

import boto3
import mlflow
from botocore.client import Config

from src import create_logger
from src.config import app_settings, setup_env

logger = create_logger("s3_verification")


def verify_s3_artifacts(
    run_id: str, expected_artifacts: list[str] | None = None
) -> dict[str, Any]:
    """
    Verifies that all expected artifacts are present in S3.

    Parameters
    ----------
    run_id : str
        MLflow run ID to verify artifacts for.
    expected_artifacts : list[str] | None
        List of expected artifacts to verify. If None, simply checks that some artifacts are present.

    Returns
    -------
    results : dict[str, Any]
        Dictionary containing the results of verification.
        "success" : bool
            Whether all expected artifacts were found.
        "artifact_uri" : str
            MLflow artifact URI.
        "s3_artifacts" : list[str]
            List of S3 artifacts found.
        "missing_artifacts" : list[str]
            List of expected artifacts that were not found.
        "errors" : list[str]
            List of errors encountered during verification.

    Notes
    -----
    This function assumes that the artifact URI is an S3 URI. If not, it will return an error.
    """
    results: dict[str, Any] = {
        "success": False,
        "artifact_uri": "",
        "s3_artifacts": [],
        "missing_artifacts": [],
        "errors": [],
    }

    try:
        # Get artifact URI from MLflow
        client = mlflow.tracking.MlflowClient()  # type: ignore
        run = client.get_run(run_id)
        artifact_uri: str = run.info.artifact_uri
        results["artifact_uri"] = artifact_uri

        # Check if artifact URI is S3
        if not artifact_uri.startswith("s3://"):
            results["errors"].append(f"Artifact URI is not S3: {artifact_uri}")
            return results

        # Parse S3 URI
        # Format: s3://bucket/path/to/artifacts
        parts = artifact_uri.replace("s3://", "").split("/", 1)
        bucket = parts[0]
        prefix = parts[1] if len(parts) > 1 else ""

        # Ensure environment variables are set
        setup_env()
        s3_client = boto3.client(
            "s3",
            endpoint_url=app_settings.mlflow_s3_endpoint_url,
            config=Config(signature_version="s3v4"),
        )

        # List objects in S3
        response: dict[str, Any] = s3_client.list_objects_v2(
            Bucket=bucket, Prefix=prefix
        )

        if "Contents" in response:
            # Extract relative paths
            s3_objects = []
            for obj in response["Contents"]:
                relative_path = obj["Key"].replace(prefix, "").lstrip("/")
                if relative_path:  # Skip empty paths
                    s3_objects.append(relative_path)

            results["s3_artifacts"] = s3_objects

            # Check for expected artifacts
            if expected_artifacts:
                for expected in expected_artifacts:
                    found = False
                    for artifact in s3_objects:
                        if expected in artifact:
                            found = True
                            break
                    if not found:
                        results["missing_artifacts"].append(expected)

            results["success"] = (
                len(s3_objects) > 0 and len(results["missing_artifacts"]) == 0
            )

            logger.info(f"Found {len(s3_objects)} artifacts in S3 for run {run_id}")
            logger.info(f"Artifacts: {', '.join(s3_objects[:5])}...")  # Log first 5

        else:
            results["errors"].append("No artifacts found in S3")

    except Exception as e:
        results["errors"].append(str(e))
        logger.error(f"Error verifying S3 artifacts: {e}")

    return results


def log_s3_verification_results(results: dict[str, Any]) -> None:
    """
    Logs the results of S3 artifact verification to the logger.

    Parameters
    ----------
    results : dict[str, Any]
        The results of S3 artifact verification, as returned by verify_s3_artifacts.
    """
    if results["success"]:
        logger.info("✅ S3 artifact verification PASSED")
        logger.info(f"  - Artifact URI: {results['artifact_uri']}")
        logger.info(f"  - Total artifacts: {len(results['s3_artifacts'])}")
    else:
        logger.error("❌ S3 artifact verification FAILED")
        logger.error(f"  - Artifact URI: {results['artifact_uri']}")
        for error in results["errors"]:
            logger.error(f"  - Error: {error}")
        if results["missing_artifacts"]:
            logger.error(
                f"  - Missing artifacts: {', '.join(results['missing_artifacts'])}"
            )
