from datetime import datetime
from typing import Any, Literal

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
    """Result of model training."""

    run_id: str | None = Field(default=None)
    model_name: ModelType | None = Field(default=None)
    trained_model: Any = Field(alias="model")
    metrics: dict[str, Any] = Field(default_factory=dict)
    predictions: list[float] | None = Field(default=None)


class HyperparameterTuningResult(BaseSchema):
    """Result of hyperparameter tuning with Optuna."""

    run_id: str
    model_name: ModelType
    best_params: dict[str, Any] = Field(default_factory=dict)
    metrics: dict[str, float | None] = Field(default_factory=dict)
    model_uri: str
    predictions: list[float] | None = Field(default=None)


# =======================================
# ============= API SCHEMAS =============
# =======================================


class HealthCheck(BaseSchema):
    """Health check response model."""

    status: str = ""
    api_name: str = ""
    version: str = ""
    model_status: bool = False


class CapacityComponents(BaseSchema):
    """Capacity components model."""

    max_capacity: int | None = Field(
        ..., description="Maximum capacity of bikes available for rental"
    )
    current_utilization: int | None = Field(
        ..., description="Current number of bikes rented out"
    )
    available_bikes: int | None = Field(
        ..., description="Number of bikes currently available for rental"
    )
    utilization_rate: float | None = Field(
        ..., description="Current utilization rate (0 to 1)"
    )


class PriceComponents(BaseSchema):
    """Price components model."""

    price_multiplier: float = Field(..., description="Price multiplier")
    surge: float = Field(..., description="Surge amount")
    discount: float = Field(..., description="Discount amount")
    base_price: float = Field(default=..., description="Base price for bike rental")
    competitor_price: float | None = Field(default=None, description="Competitor price")
    min_price: float = Field(..., description="Minimum price")
    max_price: float = Field(..., description="Maximum price")


class PredictedPriceResponse(BaseSchema):
    """Predicted price response model."""

    status: Literal["success", "failed"] | None = Field(default=None)
    currency: Literal["NGN", "USD"] = Field(default="NGN")
    capacity_components: CapacityComponents = Field(default_factory=dict)
    price_components: PriceComponents = Field(default_factory=dict)
    final_calculations: dict[str, Any] = Field(default_factory=dict)
    timestamp: str = Field(
        default_factory=lambda: datetime.now().isoformat(timespec="seconds")
    )
    error: str | None = Field(default=None)
