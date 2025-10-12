import json
import tempfile
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Generator

import httpx
import joblib
import lightgbm as lgb
import mlflow
import narwhals as nw
import pandas as pd
import polars as pl
import xgboost as xgb
import yaml  # type: ignore
from mlflow.models.model import ModelInfo
from narwhals.typing import IntoDataFrameT

from src import create_logger
from src.exceptions import MLFlowConnectionError, MLFlowError
from src.schemas.types import ArtifactsType, ModelType

logger = create_logger("mlflow_tracker")
type WriteFn = Callable[[Any, Path], None]


def write_json(obj: dict[str, Any] | Any, filepath: Path, indent: int = 2) -> None:
    """
    Write a dictionary or any JSON-serializable object to a JSON file.

    Parameters
    ----------
    obj : dict[str, Any] | Any
        Object to serialize to JSON.
    filepath : Path
        Destination file path.
    indent : int, default=2
        JSON indentation level.

    Raises
    ------
    TypeError
        If object is not JSON serializable.
    """
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=indent)


def write_txt(obj: list[Any], filepath: Path) -> None:
    """
    Write a list of strings to a text file, each on a new line.

    Parameters
    ----------
    obj : list[Any]
        List of items to write (will be converted to strings).
    filepath : Path
        Destination file path.
    """
    with open(filepath, "w", encoding="utf-8") as f:
        for line in obj:
            f.write(f"{line}\n")


def write_yaml(obj: dict[str, Any] | Any, filepath: Path) -> None:
    """
    Write a dictionary or any YAML-serializable object to a YAML file.

    Parameters
    ----------
    obj : dict[str, Any] | Any
        Object to serialize to YAML.
    filepath : Path
        Destination file path.
    """
    with open(filepath, "w", encoding="utf-8") as f:
        yaml.dump(obj, f, default_flow_style=False, sort_keys=False)


def write_pickle(obj: Any, filepath: Path) -> None:
    """
    Write any object to a file using joblib.

    Parameters
    ----------
    obj : Any
        Object to pickle.
    filepath : Path
        Destination file path.
    """
    joblib.dump(obj, filepath)


class MLFlowTracker:
    """
    MLflow tracking utility for logging experiments, models, and artifacts.

    This class provides a simplified interface for MLflow tracking operations,
    including experiment management, run tracking, and artifact logging.

    Parameters
    ----------
    tracking_uri : str
        MLflow tracking server URI (e.g., "http://localhost:5000").
    experiment_name : str
        Name of the MLflow experiment to use or create.
    """

    def __init__(self, tracking_uri: str, experiment_name: str) -> None:
        """Initialize the MLFlowTracker with a tracking URI and experiment name."""
        self.experiment_name = experiment_name
        self.tracking_uri = tracking_uri
        if not self._test_tracking_server():
            raise MLFlowConnectionError(
                f"Cannot connect to MLflow tracking server at {tracking_uri}"
            )

        mlflow.set_tracking_uri(tracking_uri)
        logger.info(f"Set MLflow tracking URI to: {tracking_uri}")
        self._set_experiment()
        logger.info(
            f"Initialized {self.__class__.__name__} with experiment: {experiment_name}"
        )

    def __repr__(self) -> str:
        """Return string representation of the tracker."""
        return f"{self.__class__.__name__}(tracking_uri={self.tracking_uri}, experiment_name={self.experiment_name})"

    def _test_tracking_server(self) -> bool:
        try:
            response = httpx.get(f"{self.tracking_uri}/#/experiments", timeout=5.0)
            logger.debug(f"MLflow tracking server is reachable at {self.tracking_uri}")
            return response.status_code == 200

        except MLFlowError as e:
            logger.error(
                f"Failed to connect to MLflow tracking server: {e}. Are you sure it's running?"
            )
            return False

    def _set_experiment(self) -> None:
        """Set the MLflow experiment. If it doesn't exist, create it."""
        try:
            experiment = mlflow.get_experiment_by_name(self.experiment_name)  # type: ignore

            if experiment is None:
                experiment_id: str = mlflow.create_experiment(self.experiment_name)  # type: ignore
                logger.info(
                    f"Created new experiment: {self.experiment_name} (ID: {experiment_id})"
                )

            experiment = mlflow.set_experiment(self.experiment_name)
            logger.info(
                f"Set experiment to: {self.experiment_name} (ID: {experiment.experiment_id})"
            )

        except MLFlowError as e:
            logger.warning(f"Failed to set experiment {self.experiment_name}: {e}")
            logger.warning("Falling back to default experiment")

    def _get_run_name(self, run_name: str | None = None) -> str:
        """
        Generate a default run name if none is provided.

        Parameters
        ----------
        run_name : str, optional
            Custom run name. If None, generates a timestamp-based name.

        Returns
        -------
        str
            Run name to use.
        """
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
        nested : bool, default=False
            Whether the run is nested within another run.
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
        logger.info(f"Started MLflow run: {run_id} (name: {run_name})")

        try:
            yield run_id
            # Auto-end run with success when context exits normally
            self.end_run(status="FINISHED")

        except MLFlowError as e:
            logger.error(f"Exception during MLflow run {run_id}: {e}")
            # End run as failed and re-raise
            try:
                self.end_run(status="FAILED")
            except MLFlowError as end_exc:
                logger.warning(f"Failed to end run {run_id} after exception: {end_exc}")
            raise

    def log_params(self, params: dict[str, Any]) -> None:
        """
        Log parameters to MLflow.

        Parameters
        ----------
        params : dict[str, Any]
            Parameters to log. Values will be converted to strings.

        Notes
        -----
        MLflow has a limit of 500 parameters per run. Parameter values
        are truncated at 500 characters.

        """
        try:
            mlflow.log_params(params)  # type: ignore
            logger.debug(f"Logged {len(params)} parameters")
        except MLFlowError as e:
            logger.error(f"Failed to log parameters: {e}")
            raise

    def log_metrics(
        self, metrics: dict[str, float | None], step: int | None = None
    ) -> None:
        """
        Log metrics to MLflow.

        Parameters
        ----------
        metrics : dict[str, float | None]
            Metrics to log. Keys are metric names, values are numeric.
        step : int, optional
            Step at which the metrics are logged (useful for iterative training).
        """
        try:
            mlflow.log_metrics(metrics, step=step)  # type: ignore
            logger.debug(
                f"Logged {len(metrics)} metrics" + (f" at step {step}" if step else "")
            )
        except MLFlowError as e:
            logger.error(f"Failed to log metrics: {e}")
            raise

    def set_tags(self, tags: dict[str, Any]) -> None:
        """
        Set tags for the current MLflow run.

        Parameters
        ----------
        tags : dict[str, Any]
            Dictionary of tags to set. Values will be converted to strings.

        Examples
        --------
        >>> tracker.set_tags({
        ...     "model_type": "xgboost",
        ...     "dataset_version": "v2",
        ...     "author": "data_scientist"
        ... })
        """
        try:
            mlflow.set_tags(tags)  # type: ignore
            logger.debug(f"Set {len(tags)} tags")
        except MLFlowError as e:
            logger.error(f"Failed to set tags: {e}")
            raise

    def log_model(
        self,
        model: Any,
        model_name: str | ModelType,
        input_example: IntoDataFrameT | None = None,
        signature: Any | None = None,  # noqa: ARG002
        registered_model_name: str | None = None,  # noqa: ARG002
        save_format: ArtifactsType = ArtifactsType.PICKLE,
    ) -> None:
        """
        Log model to MLflow as an artifact with metadata.

        Parameters
        ----------
        model : Any
            Model object to log.
        model_name : str | ModelType
            Name for the model artifact.
        input_example : IntoDataFrameT, optional
            Example input data for the model (DataFrame-like).
        signature : Any, optional
            Model signature (currently unused, reserved for future use).
        registered_model_name : str, optional
            Name for model registry (currently unused, reserved for future use).
        save_format : ArtifactsType, default=ArtifactsType.PICKLE
            Format to save the model. Options: PICKLE, JSON (for XGBoost/LightGBM).

        Notes
        -----
        - Saves model file, metadata (YAML), and input example (JSON) as artifacts
        - For XGBoost/LightGBM models, use `save_format=ArtifactsType.JSON`
        - Models are saved in `models/{model_name}/` directory

        """
        datetime_now: str = datetime.now().isoformat(timespec="seconds")
        model_name = str(model_name)
        n_example_rows: int = 5

        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                tmpdir_path = Path(tmpdir)

                # Determine model file extension and path
                if save_format == ArtifactsType.JSON:
                    model_ext: str = "json"
                elif save_format == ArtifactsType.TXT:
                    model_ext = "txt"
                else:
                    model_ext = "pkl"

                model_path = (
                    tmpdir_path / f"{model_name}_{datetime_now}_model.{model_ext}"
                )
                input_example_path = (
                    tmpdir_path / f"{model_name}_{datetime_now}_input_example.json"
                )
                metadata_path = (
                    tmpdir_path / f"{model_name}_{datetime_now}_metadata.yaml"
                )

                # Save model based on format
                if save_format == ArtifactsType.PICKLE:
                    joblib.dump(model, model_path)
                    logger.debug(f"Saved model as pickle: {model_path.name}")

                elif save_format == ArtifactsType.JSON:
                    if not hasattr(model, "save_model"):
                        raise AttributeError(
                            f"Model {type(model).__name__} does not have a 'save_model' method. "
                            "Use ArtifactsType.PICKLE instead."
                        )
                    model.save_model(str(model_path))
                    logger.debug(f"Saved model as JSON: {model_path.name}")

                elif save_format == ArtifactsType.TXT:
                    if not hasattr(model, "save_model"):
                        raise AttributeError(
                            f"Model {type(model).__name__} does not have a 'save_model' method. "
                            "Use ArtifactsType.PICKLE instead."
                        )
                    model.save_model(str(model_path))
                    logger.debug(f"Saved model as TXT: {model_path.name}")

                else:
                    raise ValueError(f"Unsupported save format: {save_format}")

                # Process input example if provided
                input_example_df: pl.DataFrame | None = None
                if input_example is not None:
                    try:
                        input_example_df = nw.from_native(input_example).to_polars()
                    except Exception as e:
                        logger.warning(f"Failed to convert input example: {e}")

                # Create comprehensive metadata
                metadata: dict[str, Any] = {
                    "model_type": model_name,
                    "framework": type(model).__module__,
                    "class": type(model).__name__,
                    "save_format": str(save_format),
                    "timestamp": datetime_now,
                }

                # Add input example info if available
                if input_example_df is not None:
                    metadata["input_example"] = {
                        "num_rows": input_example_df.shape[0],
                        "num_cols": input_example_df.shape[1],
                        "dtypes": {
                            col: str(dtype)
                            for col, dtype in zip(
                                input_example_df.columns, input_example_df.dtypes
                            )
                        },
                    }

                # Add model-specific metadata
                if hasattr(model, "get_params"):
                    try:
                        metadata["model_params"] = model.get_params()
                    except Exception as e:
                        logger.warning(f"Failed to extract model params: {e}")

                if hasattr(model, "save_config"):
                    try:
                        metadata["model_params"] = json.loads(model.save_config())
                    except Exception as e:
                        logger.warning(f"Failed to extract model params: {e}")

                # Write metadata
                write_yaml(metadata, metadata_path)

                # Write input example
                if input_example_df is not None:
                    example_data = input_example_df.head(n_example_rows).to_dicts()
                    write_json(example_data, input_example_path)

                # Log all artifacts to MLflow
                artifact_path: str = f"models/{model_name}"
                mlflow.log_artifact(str(model_path), artifact_path=artifact_path)  # type: ignore
                mlflow.log_artifact(str(metadata_path), artifact_path=artifact_path)  # type: ignore

                if input_example_df is not None:
                    mlflow.log_artifact(  # type: ignore
                        str(input_example_path), artifact_path=artifact_path
                    )

                logger.info(f"✅ Successfully logged {model_name} model and metadata")

        except MLFlowError as e:
            logger.error(f"❌ Failed to log model {model_name}: {e}")
            raise

    def log_mlflow_artifact(
        self,
        object: Any,
        object_type: ArtifactsType,
        filename: str,
        artifact_dest: str | None = None,
    ) -> None:
        """
        Log an arbitrary object as an MLflow artifact.

        Parameters
        ----------
        object : Any
            Object to log (dict, list, or any picklable object).
        object_type : ArtifactsType
            Type of artifact (JSON, TXT, YAML, or PICKLE).
        filename : str
            Base filename for the artifact (without extension).
        artifact_dest : str, optional
            Subdirectory path within the artifact store.

        """
        # Map artifact type to write function
        write_fn_map: dict[ArtifactsType, WriteFn] = {
            ArtifactsType.JSON: write_json,
            ArtifactsType.TXT: write_txt,
            ArtifactsType.YAML: write_yaml,
            ArtifactsType.PICKLE: write_pickle,
        }

        if object_type not in write_fn_map:
            raise ValueError(f"Unsupported object type: {object_type}")

        write_fn: WriteFn = write_fn_map[object_type]

        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                # Create filename with proper extension
                tmp_path = Path(tmpdir) / f"{filename}.{object_type}"
                write_fn(object, tmp_path)
                mlflow.log_artifact(str(tmp_path), artifact_path=artifact_dest)  # type: ignore

                dest_info = f" to {artifact_dest}" if artifact_dest else ""
                logger.debug(f"Logged artifact: {tmp_path.name}{dest_info}")

        except MLFlowError as e:
            logger.error(f"Failed to log artifact {filename}: {e}")
            raise

    def log_artifact_from_path(
        self, local_path: Path, artifact: str | None = None, delete_tmp: bool = True
    ) -> None:
        """
        Log a local file or directory as an MLflow artifact.

        Parameters
        ----------
        local_path : Path
            Path to the local file or directory to log.
        artifact : str, optional
            Subdirectory path within the artifact store.
        delete_tmp : bool, default=True
            Whether to delete the local file after logging.
        """
        if not local_path.exists():
            raise FileNotFoundError(f"Local path does not exist: {local_path}")

        try:
            mlflow.log_artifact(str(local_path), artifact_path=artifact)  # type: ignore
            dest_info = f" to {artifact}" if artifact else ""
            time.sleep(1)  # Ensure artifact is fully written before deletion
            logger.debug(f"Logged artifact from path: {local_path} to {dest_info}")

            if delete_tmp:
                if local_path.is_dir():
                    # Delete contents of directory but keep the directory itself
                    for item in local_path.iterdir():
                        if item.is_dir():
                            import shutil

                            shutil.rmtree(item)
                            logger.debug(f"Deleted subdirectory: {item}")
                        else:
                            item.unlink(missing_ok=True)
                            logger.debug(f"Deleted file: {item}")
                logger.debug(f"Deleted contents of directory: {local_path}")

        except MLFlowError as e:
            logger.error(f"Failed to log artifact from path {local_path}: {e}")
            raise

    def get_artifact_uri(self, artifact_path: str) -> str:
        """Get the URI of a logged artifact."""
        return mlflow.get_artifact_uri(artifact_path)  # type: ignore

    def load_model_artifact(
        self,
        run_id: str,
        model_name: str | ModelType,
        artifact_subpath: str = "models",
    ) -> dict[str, Any]:
        """
        Load a model artifact and its metadata from MLflow.

        Parameters
        ----------
        run_id : str
            MLflow run ID containing the model.
        model_name : str | ModelType
            Name of the model (e.g., "xgboost", "lightgbm", "random_forest").
        artifact_subpath : str, default="models"
            Subdirectory path where model artifacts are stored.

        Returns
        -------
        dict[str, Any]
            Dictionary containing:
            - "model": The loaded model object
            - "metadata": Model metadata (dict)
            - "input_example": Input example data (list of dicts), if available
            - "model_uri": URI to the model artifact directory

        """
        client = mlflow.tracking.MlflowClient()  # type: ignore
        model_name = str(model_name)

        try:
            # Construct artifact path
            artifact_path: str = f"{artifact_subpath}/{model_name}"

            # Download all artifacts to a temporary directory
            with tempfile.TemporaryDirectory() as tmpdir:
                tmpdir_path = Path(tmpdir)
                local_path = client.download_artifacts(
                    run_id, artifact_path, dst_path=str(tmpdir_path)
                )
                local_path = Path(local_path)

                # Find model file (look for .pkl, .json, .txt)
                model_files = list(local_path.glob("*_model.*"))
                if not model_files:
                    raise FileNotFoundError(f"No model file found in {local_path}")

                model_file = model_files[0]
                model_ext = model_file.suffix[1:]
                logger.info(
                    f"Detected model file: {model_file.name!r} with extension {model_ext!r}"
                )

                # Load model based on extension
                model: Any
                if model_ext == ArtifactsType.PICKLE.value:
                    model = joblib.load(model_file)
                    logger.debug(f"Loaded pickle model from {model_file.name}")

                elif model_ext == ArtifactsType.JSON.value:
                    model = xgb.Booster()
                    model.load_model(str(model_file))
                    logger.debug(f"Loaded XGBoost model from {model_file.name}")

                elif model_ext == ArtifactsType.TXT.value:
                    model = lgb.Booster(model_file=str(model_file))
                    logger.debug(f"Loaded LightGBM model from {model_file.name}")

                else:
                    raise ValueError(f"Unsupported model file extension: {model_ext}")

                # Load metadata
                metadata_files = list(local_path.glob("*_metadata.yaml"))
                metadata: dict[str, Any] = {}
                if metadata_files:
                    with open(metadata_files[0]) as f:
                        metadata = yaml.safe_load(f)
                    logger.debug(f"Loaded metadata from {metadata_files[0].name}")

                # Load input example
                input_example_files = list(local_path.glob("*_input_example.json"))
                input_example: list[dict[str, Any]] | None = None
                if input_example_files:
                    with open(input_example_files[0]) as f:
                        input_example = json.load(f)
                    logger.debug(
                        f"Loaded input example from {input_example_files[0].name}"
                    )

                # Construct model URI
                model_uri = f"runs:/{run_id}/{artifact_path}"

                logger.info(
                    f"✅ Successfully loaded {model_name} model from run {run_id}"
                )

                return {
                    "model": model,
                    "metadata": metadata,
                    "input_example": input_example,
                    "model_uri": model_uri,
                    "run_id": run_id,
                    "model_name": model_name,
                }

        except Exception as e:
            logger.error(f"❌ Failed to load model artifact: {e}")
            raise

    def load_json_artifact(
        self,
        run_id: str,
        artifact_path: str,
    ) -> dict[str, Any]:
        """
        Load a JSON artifact from MLflow.

        Parameters
        ----------
        run_id : str
            MLflow run ID containing the artifact.
        artifact_path : str
            Path to the JSON artifact (e.g., "models/xgboost/feat_imp_xgboost.json").

        Returns
        -------
        dict[str, Any]
            Loaded JSON data.

        Examples
        --------
        >>> tracker = MLFlowTracker(tracking_uri="http://localhost:5001", experiment_name="my_exp")
        >>> feat_imp = tracker.load_json_artifact(
        ...     run_id="abc123",
        ...     artifact_path="models/xgboost/feat_imp_xgboost.json"
        ... )

        """
        client = mlflow.tracking.MlflowClient()  # type: ignore

        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                local_path = client.download_artifacts(
                    run_id, artifact_path, dst_path=tmpdir
                )
                with open(local_path) as f:
                    data = json.load(f)

                logger.debug(f"Loaded JSON artifact: {artifact_path}")
                return data

        except Exception as e:
            logger.error(f"Failed to load JSON artifact {artifact_path}: {e}")
            raise

    def list_artifacts(self, run_id: str, path: str | None = None) -> list[str]:
        """
        List all artifacts for a given run.

        Parameters
        ----------
        run_id : str
            MLflow run ID.
        path : str, optional
            Artifact subdirectory to list. If None, lists all artifacts.

        Returns
        -------
        list[str]
            List of artifact paths.

        Examples
        --------
        >>> tracker = MLFlowTracker(tracking_uri="http://localhost:5001", experiment_name="my_exp")
        >>> artifacts = tracker.list_artifacts(run_id="abc123")
        >>> print(artifacts)
        ['models/xgboost/xgboost_model.json', 'models/xgboost/metadata.yaml', ...]

        """
        client = mlflow.tracking.MlflowClient()  # type: ignore

        try:
            artifacts = client.list_artifacts(run_id, path)
            artifact_paths = [artifact.path for artifact in artifacts]

            logger.debug(f"Found {len(artifact_paths)} artifacts for run {run_id}")
            return artifact_paths

        except Exception as e:
            logger.error(f"Failed to list artifacts for run {run_id}: {e}")
            raise

    def register_model(
        self,
        run_id: str,
        model: Any,
        model_name: str,
        input_example: IntoDataFrameT | None,
    ) -> None:
        """Register a model with MLflow."""
        if input_example is not None:
            try:
                input_example_df: pd.DataFrame | None = (
                    nw.from_native(input_example).to_pandas().head(2)
                )
            except Exception as e:
                logger.warning(f"Failed to convert input example: {e}")

        input_example_df = (
            input_example_df.drop(columns=["target"])
            if isinstance(input_example_df, pd.DataFrame)
            else None
        )

        try:
            with mlflow.start_run(run_id=run_id):  # type: ignore
                # Determine model type
                if "XGBOOST" in model_name:
                    model_info: ModelInfo = mlflow.xgboost.log_model(
                        xgb_model=model,
                        artifact_path="registered_model",
                        registered_model_name=f"{model_name}_best",
                        input_example=input_example_df,
                    )
                elif "LIGHTGBM" in model_name:
                    model_info = mlflow.lightgbm.log_model(
                        lgb_model=model,
                        artifact_path="registered_model",
                        registered_model_name=f"{model_name}_best",
                        input_example=input_example_df,
                    )
                elif "RANDOM_FOREST" in model_name:
                    model_info = mlflow.sklearn.log_model(
                        sk_model=model,
                        artifact_path="registered_model",
                        registered_model_name=f"{model_name}_best",
                        input_example=input_example_df,
                    )
                else:
                    raise ValueError(f"Unsupported model type: {model_name}")
                logger.info(
                    f"✅ Registered model: {model_info.model_uri} | version: {model_info.registered_model_version}"
                )
                return

        except Exception as e:
            logger.error(f"❌ Failed to register model: {e}")
            raise

    def end_run(self, status: str = "FINISHED") -> None:
        """
        End the current MLflow run.

        Parameters
        ----------
        status : str, default="FINISHED"
            Status to end the run with. Options: "FINISHED", "FAILED", "KILLED".

        Notes
        -----
        - Automatically called by the context manager on normal exit
        - If status is "FINISHED", attempts to sync artifacts to S3 (if configured)
        """
        run = mlflow.active_run()
        run_id = run.info.run_id if run else None

        try:
            mlflow.end_run(status=status)  # type: ignore
            logger.info(f"Ended MLflow run with status: {status}")

            # Sync artifacts to S3 if run was successful
            if run_id and status == "FINISHED":
                try:
                    # TODO: Implement S3 sync logic here
                    # from .mlflow_s3_utils import MLflowS3Manager
                    # s3_manager = MLflowS3Manager()
                    # s3_manager.sync_mlflow_artifacts_to_s3(run_id)
                    # logger.info(f"✅ Synced artifacts to S3 for run {run_id}")
                    pass
                except MLFlowError as e:
                    logger.warning(f"❌ Failed to sync artifacts to S3: {e}")

        except MLFlowError as e:
            logger.error(f"Failed to end run: {e}")
            raise
