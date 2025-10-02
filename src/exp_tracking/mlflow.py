import json
import tempfile
from contextlib import contextmanager
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Generator

import joblib
import mlflow
import narwhals as nw
import polars as pl
import yaml  # type: ignore
from narwhals.typing import IntoDataFrameT

from src import create_logger

logger = create_logger("mlflow_tracker")
type WriteFn = Callable[[Any, Path], None]


class ArtifactsType(str, Enum):
    """Enum for different types of artifacts."""

    JSON = "json"
    TXT = "txt"
    YAML = "yaml"
    ANY = "joblib"

    def __str__(self) -> str:
        return str(self.value)


def write_json(object: dict[str, Any] | Any, filepath: Path, indent: int = 2) -> None:
    """Write a dictionary or any JSON-serializable object to a JSON file."""
    with open(filepath, "w") as f:
        json.dump(object, fp=f, indent=indent)


def write_txt(object: list[Any], filepath: Path) -> None:
    """Write a list of strings to a text file, each on a new line."""
    with open(filepath, "w") as f:
        for line in object:
            f.write(line + "\n")


def write_yaml(object: dict[str, Any] | Any, filepath: Path) -> None:
    """Write a dictionary or any YAML-serializable object to a YAML file."""
    with open(filepath, "w") as f:
        yaml.dump(object, f)


def write_pickle(object: dict[str, Any] | Any, filepath: Path) -> None:
    """Write any object to a file using joblib."""
    joblib.dump(object, filepath)


class MLFlowTracker:
    def __init__(self, tracking_uri: str, experiment_name: str) -> None:
        """
        Initialize the MLFlowTracker with a tracking URI and experiment name.
        This sets up the MLflow tracking server and experiment.
        """
        self.experiment_name = experiment_name
        self.tracking_uri = tracking_uri
        mlflow.set_tracking_uri(tracking_uri)
        self._set_experiment()
        logger.info(
            f"Initialized {self.__class__.__name__} with experiment: {experiment_name}"
        )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(tracking_uri={self.tracking_uri}, experiment_name={self.experiment_name})"

    def _set_experiment(self) -> None:
        """Set the MLflow experiment. If it doesn't exist, create it."""
        try:
            mlflow.set_experiment(self.experiment_name)

        except Exception as e:
            logger.warning(f"Failed to set experiment {self.experiment_name}: {e}")

    def _get_run_name(self, run_name: str | None = None) -> str:
        """Generate a default run name if none is provided."""
        if run_name is None:
            run_name = f"run_{datetime.now().isoformat(timespec='seconds')}"
        return run_name

    @contextmanager
    def start_run(
        self,
        run_name: str | None = None,
        nested: bool = False,
        tags: dict[str, str] | None = None,
    ) -> Generator[str, None, None]:
        """
        Context manager to start and automatically end an MLflow run.

        Parameters
        ----------
        run_name : str, optional
            Name of the run to start. If not provided, a default name will be generated.
        nested : bool, optional
            Whether the run is nested within another run. Defaults to False.
        tags : dict[str, str], optional
            Tags to be added to the run.

        Yields
        ------
        str
            ID of the started run.
        """
        run_name = self._get_run_name(run_name)
        run = mlflow.start_run(run_name=run_name, tags=tags, nested=nested)  # type: ignore
        run_id = run.info.run_id
        logger.info(f"Started MLflow run: {run_id}")

        try:
            yield run_id
            # Auto-end run with success when context exits normally
            self.end_run(status="FINISHED")

        except Exception as e:
            logger.error(f"Exception during MLflow run {run_id}: {e}")
            # End run as failed and re-raise
            try:
                self.end_run(status="FAILED")
            except Exception as end_exc:
                logger.warning(f"Failed to end run {run_id} after exception: {end_exc}")
            raise

    def log_params(self, params: dict[str, Any]) -> None:
        """
        Log parameters to MLflow.

        Parameters
        ----------
        params : dict[str, Any]
            Parameters to log.
        """
        mlflow.log_params(params)  # type: ignore

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        """
        Log metrics to MLflow.

        Parameters
        ----------
        metrics : dict[str, float]
            Metrics to log.
        step : int, optional
            Step at which the metrics are logged.
        """
        mlflow.log_metrics(metrics, step=step)  # type: ignore

    def set_tags(self, tags: dict[str, Any]) -> None:
        """
        Set tags for the current MLflow run.

        Parameters
        ----------
        tags : dict[str, str]
            Dictionary of tags to set.
        """
        mlflow.set_tags(tags)  # type: ignore

    def log_model(
        self,
        model: Any,
        model_name: str,
        input_example: IntoDataFrameT | None = None,  # noqa: ARG002
        signature: Any | None = None,  # noqa: ARG002
        registered_model_name: str | None = None,  # noqa: ARG002
    ) -> None:
        """
        Log model to MLflow with compatibility for different versions.

        Parameters
        ----------
        model : Any
            Model to log.
        model_name : str
            Name of the model to log.
        input_example : pd.DataFrame, optional
            Example input to be used for logging model signature.
        signature : Any, optional
            Model signature to be used for logging.
        registered_model_name : str, optional
            Name of the registered model to log.

        """
        datetime_now: str = datetime.now().isoformat(timespec="seconds")
        N: int = 5
        try:
            # Save model to a temporary file first
            with tempfile.TemporaryDirectory() as tmpdir:
                model_path = Path(tmpdir, f"{model_name}_{datetime_now}_model.pkl")
                input_example_path = Path(
                    tmpdir, f"{model_name}_{datetime_now}_input_example.json"
                )
                metadata_path = Path(
                    tmpdir, f"{model_name}_{datetime_now}_metadata.yaml"
                )

                # Save model
                joblib.dump(model, model_path)

                # Format input data example if provided
                if input_example is not None:
                    input_example_df: pl.DataFrame = nw.from_native(
                        input_example
                    ).to_polars()  # type: ignore
                else:
                    input_example_df = None

                # Create metadata
                metadata = {
                    "model_type": model_name,
                    "framework": type(model).__module__,
                    "class": type(model).__name__,
                    "input_example": {
                        "num_rows": input_example_df.shape[0],
                        "num_cols": input_example_df.shape[1],
                    }
                    if input_example_df is not None
                    else None,
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                }
                with open(metadata_path, "w") as f:
                    yaml.dump(metadata, f)

                if input_example_df is not None:
                    with open(input_example_path, "w") as f:
                        json.dump(input_example_df.head(N).to_dicts(), f, indent=4)

                # Log artifacts
                artifact_path: str = f"models/{model_name}"
                mlflow.log_artifact(model_path, artifact_path=artifact_path)  # type: ignore
                mlflow.log_artifact(metadata_path, artifact_path=artifact_path)  # type: ignore

                if input_example_df is not None:
                    mlflow.log_artifact(input_example_path, artifact_path=artifact_path)  # type: ignore

                logger.info(f"✅ Successfully saved {model_name} model as artifact")

        except Exception as e:
            logger.error(f"❌ Failed to log model {model_name}: {e}")

    def log_mlflow_artifact(
        self,
        object: Any,
        object_type: ArtifactsType,
        filename: str,
        artifact_dest: str | None = None,
    ) -> None:
        """
        Log a local file to MLflow.

        Parameters
        ----------
        local_path : str
            Path to the local file to log.
        artifact_dest : str | None
            (Optional) Run-relative directory in the MLflow artifact store.
        """
        if object_type == ArtifactsType.JSON:
            write_fn: WriteFn = write_json
        elif object_type == ArtifactsType.TXT:
            write_fn = write_txt
        elif object_type == ArtifactsType.YAML:
            write_fn = write_yaml
        elif object_type == ArtifactsType.ANY:
            write_fn = write_pickle
        else:
            raise ValueError(f"Unsupported object type: {object_type}")

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir) / f"{filename}-artifact.{object_type}"
            write_fn(object, tmp_path)
            mlflow.log_artifact(tmp_path, artifact_path=artifact_dest)  # type: ignore

    def end_run(self, status: str = "FINISHED") -> None:
        """
        End the current MLflow run.

        Parameters
        ----------
        status : str
            Status to end the run with. Defaults to "FINISHED".

        Notes
        -----
        If the run is successful, the artifacts will be synced to S3.
        """
        # Get run ID before ending
        run = mlflow.active_run()
        run_id = run.info.run_id if run else None

        mlflow.end_run(status=status)  # type: ignore
        logger.info("🚨 Ended MLflow run")

        # Sync artifacts to S3 after run ends
        if run_id and status == "FINISHED":
            try:
                # TODO: Implement S3 sync logic here
                # Logic to sync artifacts to S3

                # from .mlflow_s3_utils import MLflowS3Manager
                # s3_manager = MLflowS3Manager()
                # s3_manager.sync_mlflow_artifacts_to_s3(run_id)

                logger.info(f"✅ Synced artifacts to S3 for run {run_id}")

            except Exception as e:
                logger.warning(f"❌ Failed to sync artifacts to S3: {e}")
