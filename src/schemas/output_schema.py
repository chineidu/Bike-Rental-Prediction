from typing import Any

from pydantic import Field

from src.schemas.types import ModelType

from .input_schema import BaseSchema, Float


class DataSchema(BaseSchema):
    numeric: dict[str, str] = Field(default_factory=dict)
    categorical: dict[str, str] = Field(default_factory=dict)


class DataShape(BaseSchema):
    total_rows: int
    total_columns: int
    number_of_numeric_columns: int
    number_of_categorical_columns: int


class NumericColumnStats(BaseSchema):
    column: str
    mean: Float | None = None
    median: Float | None = None
    mode: list[Float | None] = Field(default_factory=list)
    std: Float | None = None
    variance: Float | None = None
    range: Float | None = None
    min: Float | None = None
    max: Float | None = None
    count: int | None = None
    missing_values: int | None = None
    missing_pct: Float | None = None
    unique_values: int | None = None


class CategoricalColumnStats(BaseSchema):
    column: str
    total_count: int | None = None
    unique_values: int | None = None
    value_counts: list[tuple[str, int]] = Field(default_factory=list)
    missing_values: int | None = None
    missing_pct: Float | None = None


class SummaryStatistics(BaseSchema):
    numeric: list[NumericColumnStats] = Field(default_factory=list)
    categorical: list[CategoricalColumnStats] = Field(default_factory=list)


class OtherInfo(BaseSchema):
    num_unique_numeric_rows: dict[str, int] = Field(default_factory=dict)
    num_unique_categorical_rows: dict[str, int] = Field(default_factory=dict)
    data_nulls: dict[str, int] = Field(default_factory=dict)
    total_nulls: int | None = None
    num_duplicated_rows: int | None = None
    memory_usage_MB: Float | None = None  # noqa: N815
    validation_timestamp: str | None = None


class DataValidatorSchema(BaseSchema):
    data_schema: DataSchema
    data_shape: DataShape
    summary_statistics: SummaryStatistics
    other_info: OtherInfo


class TrainingResult(BaseSchema):
    """
    Result of model training.

    Attributes
    ----------
    run_id : str | None
        MLflow run ID for the training experiment.
    model_name : ModelType | None
        Type of model trained (xgboost, lightgbm, random_forest).
    trained_model : Any
        Trained model object. Can be:
        - sklearn models (RandomForestRegressor, etc.)
        - XGBoost models (xgb.XGBRegressor, xgb.Booster)
        - LightGBM models (lgb.LGBMRegressor, lgb.Booster)
        - Any other trained ML model
    metrics : dict[str, Any]
        Evaluation metrics (RMSE, MAE, MAPE, R2, etc.).
    predictions : list[float] | None
        List of predictions on the test set.
    """

    run_id: str | None = Field(default=None)
    model_name: ModelType | None = Field(default=None)
    trained_model: Any = Field(alias="model")
    metrics: dict[str, Any] = Field(default_factory=dict)
    predictions: list[float] | None = Field(default=None)


class HyperparameterTuningResult(BaseSchema):
    """
    Result of hyperparameter tuning with Optuna.

    Attributes
    ----------
    run_id : str
        MLflow run ID for the hyperparameter tuning experiment.
    model_name : ModelType
        Type of model that was tuned (xgboost, lightgbm, random_forest).
    best_params : dict[str, Any]
        Best hyperparameters found during optimization.
    metrics : dict[str, float | None]
        Evaluation metrics (best_rmse, mean_rmse, mean_mae, mean_mape, mean_adjusted_r2).
    model_uri : str
        MLflow URI to the best model artifact.
    """

    run_id: str
    model_name: ModelType
    best_params: dict[str, Any] = Field(default_factory=dict)
    metrics: dict[str, float | None] = Field(default_factory=dict)
    model_uri: str


# ============ API SCHEMAS ============ #
class HealthCheck(BaseSchema):
    """
    Health check response model.
    """

    status: str = "healthy"
    version: str = "0.1.0"
