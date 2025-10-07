import traceback
from pathlib import Path
from typing import Any

import pendulum
from airflow.providers.standard.operators.bash import BashOperator
from airflow.sdk import Asset, Param, dag, task
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
    "artifacts_path": Param(
        default=str(ARTIFACTS_DIR),
        description="Path to the artifacts directory",
    ),
    "data_path": Param(
        default="/opt/data/database.parquet",
        description="Path to the training data Parquet file",
    ),
    "features_path": Param(
        default=str(ARTIFACTS_DIR / "features" / "transformed_features.parquet"),
        description="Path to the features Parquet file",
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
        try:
            artifacts_path: str = (
                params["artifacts_path"] if params else "/opt/airflow/artifacts/"
            )

            artifacts_dir = Path(artifacts_path)
            # Create parent directory if it doesn't exist
            artifacts_dir.mkdir(parents=True, exist_ok=True)

            features_dir = artifacts_dir / "features"
            models_dir = artifacts_dir / "models"
            reports_dir = artifacts_dir / "reports"

            # Create directories if they don't exist
            features_dir.mkdir(parents=True, exist_ok=True)
            models_dir.mkdir(parents=True, exist_ok=True)
            reports_dir.mkdir(parents=True, exist_ok=True)

            print(f"✅ Artifacts directory initialized at: {artifacts_dir}")

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
        import polars as pl

        from src.utilities.data_validator import data_validator

        try:
            data_path: str | None = result.get("data_path")
            data: pl.DataFrame = pl.read_parquet(data_path)
            validation_result = data_validator(data).model_dump()

            return {
                "validation_result": validation_result,
                "data_path": result.get("data_path"),
                "datetime": get_time_now(),
            }
        except Exception as e:
            print(f"❌ Data validation failed: {e}")
            raise

    @task(multiple_outputs=True, retries=2)
    def features_generation_task(
        result: dict[str, Any], params: dict[str, Any]
    ) -> dict[str, Any]:
        import polars as pl

        from src.config import app_config
        from src.ml.feature_engineering import FeatureEngineer

        try:
            save_path: str = params["features_path"]
            data_path: str = result["data_path"]
            data: pl.DataFrame = pl.read_parquet(data_path)
            feature_eng = FeatureEngineer()
            features_df: pl.DataFrame = feature_eng.create_all_features(
                data=data, config=app_config.feature_config
            )  # type: ignore

            print("🚨 Saving generated features ...")
            features_df.write_parquet(file=save_path)

            return {
                "num_rows": features_df.height,
                "num_columns": features_df.width,
                "features_path": save_path,
                "datetime": get_time_now(),
            }
        except Exception as e:
            print(f"❌ Feature generation failed: {e}")
            raise

    @task(multiple_outputs=True, retries=2)
    def model_training_task(
        result: dict[str, Any], params: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        import polars as pl

        from src.ml.trainer import ModelTrainer

        tune_models: bool = params["tune_models"] if params else False
        serializable_results: list[dict[str, Any]] = []

        data_path: str = result["features_path"]
        features_df: pl.DataFrame = pl.read_parquet(data_path)
        trainer = ModelTrainer(features_df)
        run_id: str = ""

        if not tune_models:
            print("🚨 Training models without hyperparameter tuning ...")
            try:
                training_results: list[dict[str, Any]] = trainer.train_all_models()

                for row in training_results:
                    serializable_results.append(  # noqa: PERF401
                        {
                            "run_id": row.get("run_id", run_id),
                            "model_name": row.get("model_name", ""),
                            "metrics": row.get("metrics", {}),
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
        import polars as pl

        from src.ml.trainer import ModelTrainer

        data_path: str = result["features_path"]
        tune_models: bool = params["tune_models"] if params else False
        features_df: pl.DataFrame = pl.read_parquet(data_path)
        trainer = ModelTrainer(features_df)

        if tune_models:
            print("🚨 Performing hyperparameter tuning ...")
            try:
                hyperparams_results: list[dict[str, Any]] = (
                    trainer.hyperparameter_tuning_all_models()
                )

                return {
                    "hyperparameter_tuning_results": hyperparams_results,
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

    _ = BashOperator(
        task_id="bash_task",
        bash_command="""
        echo "Cleaning up temporary files..."
        rm -rf /remove_temp_files/ || true
        """,
        trigger_rule="all_done",  # Ensure this runs even if upstream tasks fail
    )

    result_load = load_data_task()
    result_init_artifacts = init_artifacts_store()
    result_validate = validate_data_task(result_load)
    result_features = features_generation_task(result_load)
    result_train = model_training_task(result_features)
    result_tune = hyperparameter_tuning_task(result_features)

    # Branching logic
    tuning_decision = decide_tuning_branch()

    # Set task dependencies
    result_init_artifacts >> [result_load, result_validate]
    result_validate >> result_features >> tuning_decision
    tuning_decision >> [result_tune, result_train]


bike_rental_ml_dag = ml_pipeline_dag()
