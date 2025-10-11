from pydantic import Field

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


class HealthCheck(BaseSchema):
    """
    Health check response model.
    """

    status: str = "healthy"
    version: str = "0.1.0"
