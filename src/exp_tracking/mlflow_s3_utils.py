"""This module provides utilities for managing MLflow artifacts in S3/MinIO.
Copied from: https://github.com/airscholar/astro-salesforecast/blob/main/include/utils/mlflow_s3_utils.py
"""

import os
import shutil

import boto3
import mlflow
from botocore.client import Config

from src import create_logger
from src.config import app_settings, setup_env
from src.utilities.service_discovery import get_minio_endpoint

logger = create_logger("mlflow_s3_utils")


class MLflowS3Manager:
    """Manager class to ensure MLflow artifacts are stored in S3/MinIO"""

    def __init__(self) -> None:
        # Ensure environment variables are set
        setup_env()

        self.s3_client = boto3.client(
            "s3",
            endpoint_url=get_minio_endpoint(),
            aws_access_key_id=app_settings.AWS_ACCESS_KEY_ID,
            aws_secret_access_key=app_settings.AWS_SECRET_ACCESS_KEY.get_secret_value(),
            config=Config(signature_version="s3v4"),
            region_name=app_settings.AWS_DEFAULT_REGION,
        )
        self.bucket_name: str = app_settings.AWS_S3_BUCKET

    def upload_artifact_to_s3(
        self, local_path: str, run_id: str, artifact_path: str | None = None
    ) -> str:
        """
        Uploads an artifact to S3 and returns the S3 key.

        Parameters
        ----------
        local_path : str
            Local path to the artifact to upload.
        run_id : str
            MLflow run ID to use for constructing the S3 key.
        artifact_path : str | None
            Optional artifact path to use for constructing the S3 key.

        Returns
        -------
        s3_key : str
            S3 key of the uploaded artifact.
        """
        try:
            # Construct S3 key
            if artifact_path:
                s3_key: str = f"{run_id[:2]}/{run_id[2:4]}/{run_id}/artifacts/{artifact_path}/{os.path.basename(local_path)}"
            else:
                s3_key = f"{run_id[:2]}/{run_id[2:4]}/{run_id}/artifacts/{os.path.basename(local_path)}"

            # Upload to S3
            self.s3_client.upload_file(local_path, self.bucket_name, s3_key)
            logger.info(f"Uploaded {local_path} to s3://{self.bucket_name}/{s3_key}")

            return s3_key

        except Exception as e:
            logger.error(f"Failed to upload artifact to S3: {e}")
            raise

    def log_artifact_with_s3(
        self, local_path: str, artifact_path: str | None = None
    ) -> None:
        """
        Logs an artifact to MLflow and uploads it to S3.

        Parameters
        ----------
        local_path : str
            Local path to the artifact to log.
        artifact_path : str | None
            Optional artifact path to use for constructing the S3 key.
        """
        # First log to MLflow normally
        if artifact_path:
            mlflow.log_artifact(local_path, artifact_path)  # type: ignore
        else:
            mlflow.log_artifact(local_path)  # type: ignore

        # Then ensure it's in S3
        run = mlflow.active_run()
        if run:
            self.upload_artifact_to_s3(local_path, run.info.run_id, artifact_path)

    def sync_mlflow_artifacts_to_s3(self, run_id: str) -> None:
        """
        Syncs MLflow artifacts to S3.

        Parameters
        ----------
        run_id : str
            MLflow run ID to sync artifacts for.

        Notes
        -----
        This function downloads all artifacts associated with the specified MLflow run
        locally, then uploads each file to S3 under the key
        `<run_id[:2]>/<run_id[2:4]>/<run_id>/artifacts/<relative_path>`.
        """
        try:
            client = mlflow.tracking.MlflowClient()  # type: ignore

            # Download all artifacts locally
            local_dir = f"/tmp/mlflow_sync/{run_id}"
            if os.path.exists(local_dir):
                shutil.rmtree(local_dir)

            artifacts_dir = client.download_artifacts(run_id, "", dst_path=local_dir)

            # Upload each file to S3
            for root, _, files in os.walk(artifacts_dir):
                for file in files:
                    local_file = os.path.join(root, file)
                    # Calculate relative path
                    relative_path = os.path.relpath(local_file, artifacts_dir)

                    # Upload to S3
                    s3_key = (
                        f"{run_id[:2]}/{run_id[2:4]}/{run_id}/artifacts/{relative_path}"
                    )
                    self.s3_client.upload_file(local_file, self.bucket_name, s3_key)
                    logger.info(f"Synced {relative_path} to S3")

            # Clean up temp directory
            shutil.rmtree(local_dir)

            logger.info(f"Successfully synced all artifacts for run {run_id} to S3")

        except Exception as e:
            logger.error(f"Failed to sync artifacts to S3: {e}")
            raise

    def list_s3_artifacts(self, run_id: str) -> list[str]:
        """
        Lists all S3 artifacts for a given MLflow run.

        Parameters
        ----------
        run_id : str
            MLflow run ID to list artifacts for.

        Returns
        -------
        list
            List of S3 keys for all artifacts associated with the specified run.
        """
        try:
            prefix: str = f"{run_id[:2]}/{run_id[2:4]}/{run_id}/"
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket_name, Prefix=prefix
            )

            if "Contents" in response:
                return [obj["Key"] for obj in response["Contents"]]
            return []

        except Exception as e:
            logger.error(f"Failed to list S3 artifacts: {e}")
            return []
