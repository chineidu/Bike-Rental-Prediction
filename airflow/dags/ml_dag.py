import traceback
from pathlib import Path
from typing import Any

import pendulum
from airflow.providers.standard.operators.bash import BashOperator
from airflow.sdk import Asset, Param, dag, task
from airflow.sdk.definitions.xcom_arg import XComArg
from pendulum import datetime

TRAINING_DATA_ASSET = Asset("file://data/")
MODEL_REGISTRY_ASSET = Asset("file://artifacts/models/")
PERFORMANCE_REPORT_ASSET = Asset("file://artifacts/reports/")

ARTIFACTS_DIR = Path("/opt/airflow/artifacts/")

default_args: dict[str, Any] = {
    "owner": "data-team",
    "depends_on_past": False,
    "start_date": datetime(2024, 1, 1),
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": pendulum.duration(minutes=5),
}

dag_params: dict[str, Param] = {
    # Paths
    "artifacts_path": Param(
        default=str(ARTIFACTS_DIR),
        description="Path to the artifacts directory",
    ),
    "data_path": Param(
        default="/opt/data/database.parquet",
        description="Path to the training data Parquet file",
    ),
    "train_data_path": Param(
        default=str(ARTIFACTS_DIR / "data" / "train_data.parquet"),
        description="Path to the training data Parquet file",
    ),
    "val_data_path": Param(
        default=str(ARTIFACTS_DIR / "data" / "val_data.parquet"),
        description="Path to the validation data Parquet file",
    ),
    "test_data_path": Param(
        default=str(ARTIFACTS_DIR / "data" / "test_data.parquet"),
        description="Path to the test data Parquet file",
    ),
    "train_features_path": Param(
        default=str(ARTIFACTS_DIR / "features" / "transformed_train_features.parquet"),
        description="Path to the training features Parquet file",
    ),
    "val_features_path": Param(
        default=str(ARTIFACTS_DIR / "features" / "transformed_val_features.parquet"),
        description="Path to the validation features Parquet file",
    ),
    "test_features_path": Param(
        default=str(ARTIFACTS_DIR / "features" / "transformed_test_features.parquet"),
        description="Path to the test features Parquet file",
    ),
    "features_path": Param(
        default=str(ARTIFACTS_DIR / "features" / "transformed_features.parquet"),
        description="Path to the features Parquet file",
    ),
    # Data split proportions
    "val_size": Param(
        default=0.1,
        description="Proportion of data to use for validation set",
    ),
    "test_size": Param(
        default=0.1,
        description="Proportion of data to use for test set",
    ),
    # Model training
    "target_col": Param(
        default="target",
        description="Name of the target column in the dataset",
    ),
    "tune_models": Param(
        default=False,
        description="Whether to perform hyperparameter tuning for models",
    ),
}


def get_time_now() -> str:
    """Get the current time formatted as a string."""
    return pendulum.now().format("YYYY-MM-DDTHH:mm:ss")


@dag(  # type: ignore
    schedule="@weekly",
    start_date=datetime(2024, 1, 1),
    catchup=False,
    default_args=default_args,
    params=dag_params,
    description="DAG for training and evaluating bike rental prediction models",
    tags=["ml", "training", "bike-rental"],
)
def ml_pipeline_dag() -> None:
    """
    DAG for training and evaluating bike rental prediction models.

    This DAG performs the following steps:
    1. Data Preprocessing: Cleans and preprocesses the raw training data.
    2. Model Training: Trains multiple machine learning models on the preprocessed data.
    3. Model Evaluation: Evaluates the trained models and generates performance reports.
    4. Model Registration: Registers the best-performing model in the model registry.

    Parameters
    ----------
    data_path : str
        Path to the training data CSV file.

    Assets
    ------
    - TRAINING_DATA_ASSET: Asset representing the training data file.
    - MODEL_REGISTRY_ASSET: Asset representing the model registry directory.
    - PERFORMANCE_REPORT_ASSET: Asset representing the performance reports directory.
    """

    @task(multiple_outputs=True)
    def load_data_task(params: dict[str, Any] | None = None) -> dict[str, Any]:
        """Load the training data from a Parquet file."""
        # NB: The return value MUST be JSON serializable
        import polars as pl

        data_path: str = params["data_path"] if params else "/opt/data/database.parquet"
        data: pl.DataFrame = pl.read_parquet(data_path)

        return {
            "data_sample": data.to_dicts()[:10],
            "data_path": data_path,
            "datetime": get_time_now(),
        }

    @task(multiple_outputs=True, retries=2)
    def init_artifacts_store(params: dict[str, Any] | None = None) -> dict[str, Any]:
        """Initialize the artifacts directory structure."""
        try:
            artifacts_path: str = (
                params["artifacts_path"] if params else "/opt/airflow/artifacts/"
            )

            artifacts_dir = Path(artifacts_path)
            # Create parent directory if it doesn't exist
            artifacts_dir.mkdir(parents=True, exist_ok=True)

            data_dir = artifacts_dir / "data"
            features_dir = artifacts_dir / "features"
            models_dir = artifacts_dir / "models"
            reports_dir = artifacts_dir / "reports"

            # Create directories if they don't exist
            data_dir.mkdir(parents=True, exist_ok=True)
            features_dir.mkdir(parents=True, exist_ok=True)
            models_dir.mkdir(parents=True, exist_ok=True)
            reports_dir.mkdir(parents=True, exist_ok=True)

            print(
                f"✅ Artifacts directory initialized at: {artifacts_dir}\n "
                f"Created subdirectories: {data_dir}, {features_dir}, {models_dir}, {reports_dir}\n\n"
            )

            return {
                "artifacts_path": str(artifacts_dir),
                "models_path": str(models_dir),
                "reports_path": str(reports_dir),
                "datetime": get_time_now(),
            }
        except Exception as e:
            print(f"❌ Failed to initialize artifacts directory: {e}")
            raise

    @task(multiple_outputs=True, retries=2)
    def validate_data_task(result: dict[str, Any]) -> dict[str, Any]:
        """Validate the loaded data."""
        import polars as pl

        from src.utilities.data_validator import data_validator

        try:
            data_path: str | None = result.get("data_path")
            data: pl.DataFrame = pl.read_parquet(data_path)
            validation_result: dict[str, Any] = data_validator(data).model_dump()

            return {
                "validation_result": validation_result,
                "data_path": result.get("data_path"),
                "datetime": get_time_now(),
            }
        except Exception as e:
            print(f"❌ Data validation failed: {e}")
            raise

    @task(multiple_outputs=True, retries=2)
    def data_split_task(
        result: dict[str, Any], params: dict[str, Any]
    ) -> dict[str, Any]:
        """Split the data into training, validation, and test sets."""
        import polars as pl

        from src.ml.utils import split_temporal_data_to_train_val_test

        try:
            data_path: str = result["data_path"]
            test_df_size: float = params["test_size"]
            val_df_size: float = params["val_size"]
            train_data_path: str = params["train_data_path"]
            val_data_path: str = params["val_data_path"]
            test_data_path: str = params["test_data_path"]

            data: pl.DataFrame = pl.read_parquet(data_path)
            (train_df, val_df, test_df) = split_temporal_data_to_train_val_test(
                data=data,
                test_size=test_df_size,
                val_size=val_df_size,
            )
            print("🚨 Saving split datasets ...")
            train_df.write_parquet(file=train_data_path)
            val_df.write_parquet(file=val_data_path)
            test_df.write_parquet(file=test_data_path)

            return {
                "num_train_rows": train_df.height,
                "num_val_rows": val_df.height,
                "num_test_rows": test_df.height,
                "train_data_path": train_data_path,
                "val_data_path": val_data_path,
                "test_data_path": test_data_path,
                "datetime": get_time_now(),
            }
        except Exception as e:
            print(f"❌ Data splitting failed: {e}")
            raise

    @task(multiple_outputs=True, retries=2)
    def features_generation_task(
        result: dict[str, Any], params: dict[str, Any]
    ) -> dict[str, Any]:
        """Generate features from the raw data."""
        import polars as pl

        from src.config import app_config
        from src.ml.feature_engineering import FeatureEngineer

        try:
            train_data_path: str = result["train_data_path"]
            val_data_path: str = result["val_data_path"]
            test_data_path: str = result["test_data_path"]
            train_features_path: str = params["train_features_path"]
            val_features_path: str = params["val_features_path"]
            test_features_path: str = params["test_features_path"]

            train_df: pl.DataFrame = pl.read_parquet(train_data_path)
            val_df: pl.DataFrame = pl.read_parquet(val_data_path)
            test_df: pl.DataFrame = pl.read_parquet(test_data_path)
            feature_eng = FeatureEngineer()

            train_features_df: pl.DataFrame = feature_eng.create_all_features(
                data=train_df, config=app_config.feature_config
            )
            val_features_df: pl.DataFrame = feature_eng.create_all_features(
                data=val_df, config=app_config.feature_config
            )
            test_features_df: pl.DataFrame = feature_eng.create_all_features(
                data=test_df, config=app_config.feature_config
            )
            print("🚨 Saving generated features ...")
            train_features_df.write_parquet(file=train_features_path)
            val_features_df.write_parquet(file=val_features_path)
            test_features_df.write_parquet(file=test_features_path)

            return {
                "train_features_path": train_features_path,
                "val_features_path": val_features_path,
                "test_features_path": test_features_path,
                "train_features": train_features_df.head(2).to_dicts(),
                "val_features": val_df.head(2).to_dicts(),
                "test_features": test_df.head(2).to_dicts(),
                "datetime": get_time_now(),
            }
        except Exception as e:
            print(f"❌ Feature generation failed: {e}")
            raise

    @task(multiple_outputs=True, retries=2)
    def model_training_task(
        result: dict[str, Any], params: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Train machine learning models on the training data."""
        import polars as pl

        from src.ml.trainer import ModelTrainer
        from src.schemas.output_schema import TrainingResult

        tune_models: bool = params["tune_models"] if params else False
        target_col: str = params["target_col"] if params else "target"
        train_features_path: str = result["train_features_path"]
        val_features_path: str = result["val_features_path"]

        train_feats_df: pl.DataFrame = pl.read_parquet(train_features_path)
        val_feats_df: pl.DataFrame = pl.read_parquet(val_features_path)
        serializable_results: list[dict[str, Any]] = []

        trainer = ModelTrainer(
            train_data=train_feats_df, val_data=val_feats_df, target_col=target_col
        )

        if not tune_models:
            print("🚨 Training models without hyperparameter tuning ...")
            try:
                training_results: list[TrainingResult] = trainer.train_all_models()

                for row in training_results:
                    serializable_results.append(  # noqa: PERF401
                        {
                            "run_id": row.run_id,
                            "model_name": row.model_name,
                            "metrics": row.metrics,
                        }
                    )

                return {
                    "training_results": serializable_results,
                    "datetime": get_time_now(),
                }

            except Exception:
                print(traceback.format_exc())
                raise

        return {
            "training_results": [],
            "datetime": get_time_now(),
        }

    @task(multiple_outputs=True, retries=2)
    def hyperparameter_tuning_task(
        result: dict[str, Any], params: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Perform hyperparameter tuning for models."""
        import polars as pl

        from src.ml.trainer import ModelTrainer
        from src.schemas.output_schema import HyperparameterTuningResult

        tune_models: bool = params["tune_models"] if params else False
        target_col: str = params["target_col"] if params else "target"
        train_features_path: str = result["train_features_path"]
        val_features_path: str = result["val_features_path"]

        train_feats_df: pl.DataFrame = pl.read_parquet(train_features_path)
        val_feats_df: pl.DataFrame = pl.read_parquet(val_features_path)

        trainer = ModelTrainer(
            train_data=train_feats_df, val_data=val_feats_df, target_col=target_col
        )

        if tune_models:
            print("🚨 Performing hyperparameter tuning ...")
            try:
                hyperparams_results: list[HyperparameterTuningResult] = (
                    trainer.hyperparameter_tuning_all_models()
                )

                return {
                    "hyperparameter_tuning_results": [
                        row.model_dump() for row in hyperparams_results
                    ],
                    "datetime": get_time_now(),
                }

            except Exception:
                print(traceback.format_exc())
                raise

        return {
            "hyperparameter_tuning_results": [],
            "datetime": get_time_now(),
        }

    @task.branch
    def decide_tuning_branch(params: dict[str, Any] | None = None) -> str:
        tune_models: bool = params["tune_models"] if params else False
        if tune_models:
            print("✅ Hyperparameter tuning enabled - branching to tuning task")
            return "hyperparameter_tuning_task"

        print("🚨 Hyperparameter tuning disabled - skipping tuning task")
        return "model_training_task"

    @task(multiple_outputs=True, retries=2, trigger_rule="none_failed_min_one_success")
    def register_best_model_task(
        result: dict[str, Any], params: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Register the best model from the experiment.

        "none_failed_min_one_success": Run this task if at least one upstream task succeeded
        "all_success": Run only if ALL upstream tasks succeed (default)
        """
        import os

        import polars as pl

        from src.config import app_config
        from src.exp_tracking.model_loader import load_best_model
        from src.ml.trainer import ModelTrainer

        target_col: str = params["target_col"] if params else "target"
        train_features_path: str = result["train_features_path"]
        val_features_path: str = result["val_features_path"]

        train_feats_df: pl.DataFrame = pl.read_parquet(train_features_path)
        val_feats_df: pl.DataFrame = pl.read_parquet(val_features_path)

        trainer = ModelTrainer(
            train_data=train_feats_df, val_data=val_feats_df, target_col=target_col
        )

        try:
            # Get MLflow tracking URI from environment variable or use default
            mlflow_uri: str = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5001")
            print(f"🔗 Using MLflow tracking URI: {mlflow_uri}\n\n")

            # Load the best model from the experiment
            best_model_artifacts: dict[str, Any] | None = load_best_model(
                experiment_name=app_config.experiment_config.experiment_name,
                tracking_uri=mlflow_uri,
            )
            if best_model_artifacts:
                trainer.mlflow_tracker.register_model(
                    run_id=best_model_artifacts["run_id"],
                    model=best_model_artifacts["model"],
                    model_name=best_model_artifacts["model_name"],
                    input_example=train_feats_df.drop([target_col]).head(5),
                )
            return {
                "status": "✅ Model registered successfully",
                "datetime": get_time_now(),
            }

        except Exception:
            print(traceback.format_exc())
            raise

    cleanup = BashOperator(
        task_id="cleanup_task",
        bash_command="""
        echo "⏳ Starting cleanup process..."

        # Remove temporary Python cache files
        echo "  → Removing __pycache__ directories..."
        find /opt/airflow -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
        find /opt/airflow -type f -name "*.pyc" -delete 2>/dev/null || true
        find /opt/airflow -type f -name "*.pyo" -delete 2>/dev/null || true

        # Clean up old MLflow artifacts (older than 7 days)
        echo "  → Cleaning old MLflow artifacts..."
        find /opt/airflow/artifacts -type f -mtime +7 -delete 2>/dev/null || true

        # Remove temporary directories
        echo "  → Removing temporary directories..."
        rm -rf /tmp/mlflow-* /tmp/tmp* /tmp/airflow-* 2>/dev/null || true

        # Clean up pip cache
        echo "  → Clearing pip cache..."
        pip cache purge 2>/dev/null || true

        # Display disk usage
        echo ""
        echo "📊 Current disk usage:"
        df -h /opt/airflow/artifacts 2>/dev/null || true

        echo ""
        echo "✅ Cleanup completed successfully!"
        """,
        trigger_rule="all_done",  # Ensure this runs even if upstream tasks fail
    )

    result_load: XComArg = load_data_task()
    result_init_artifacts: XComArg = init_artifacts_store()
    result_validate: XComArg = validate_data_task(result_load)
    result_data_split: XComArg = data_split_task(result_load)
    result_features: XComArg = features_generation_task(result_data_split)
    result_train: XComArg = model_training_task(result_features)
    result_tune: XComArg = hyperparameter_tuning_task(result_features)
    result_register: XComArg = register_best_model_task(result_features)

    # Branching logic
    tuning_decision: XComArg = decide_tuning_branch()

    # Set task dependencies
    result_init_artifacts >> [result_load, result_validate]
    result_validate >> [result_data_split, result_features] >> tuning_decision
    tuning_decision >> [result_tune, result_train] >> result_register

    # Run cleanup after model registration completes (success or failure)
    result_register >> cleanup


bike_rental_ml_dag = ml_pipeline_dag()
