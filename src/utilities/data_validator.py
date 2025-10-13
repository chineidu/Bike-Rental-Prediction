from dataclasses import dataclass, field
from typing import Any, Callable

import narwhals as nw
import narwhals.selectors as n_cs
import pendulum
from narwhals.typing import IntoFrameT

from src.exceptions import DataValidationError
from src.schemas import DataValidatorSchema

EMPTY_DATAFRAME: str = "🚫 Empty dataframe"

type InfoFn = Callable[[nw.DataFrame], dict[str, Any]]
type SchemaFn = Callable[[nw.DataFrame], dict[str, Any]]
type SummaryStatsFn = Callable[[nw.DataFrame], list[dict[str, Any]]]


@dataclass
class DataValidatorConfig:
    """
    Configuration container for data validation routines.

    Parameters
    ----------
    data : IntoFrameT
        The input data to validate. Expected to be or convertible to a frame-like
        object (e.g., pandas, polars and pyarrow table) as accepted by the validation utilities.
    schema_fns : list[SchemaFn], optional
        A list of schema-producing callables. By convention this should contain
        the numeric schema function followed by the categorical schema function,
        in that order.
    info_fns : list[InfoFn], optional
        A list of informational functions that compute or extract additional
        metadata from the data.
    summary_fn : list[SummaryStatsFn], optional
        A list of functions that compute summary statistics for the dataset.
        These functions should return summary-level metrics.

    Notes
    -----
    - The dataclass does not itself perform validation; it is a lightweight
    configuration object passed into validator routines.
    - The exact callable signatures for SchemaFn, InfoFn, and SummaryStatsFn
    should be defined elsewhere in the utilities module and adhered to by any
    provided functions.
    """

    data: IntoFrameT
    # Contains numeric and categorical schema functions in that order
    schema_fns: list[SchemaFn] = field(default_factory=list)
    info_fns: list[InfoFn] = field(default_factory=list)
    summary_fns: list[SummaryStatsFn] = field(default_factory=list)


def get_numeric_summary_stats(data: nw.DataFrame) -> list[dict[str, Any]]:
    """
    Compute summary statistics for numeric columns in a DataFrame.

    Parameters
    ----------
    data : nw.DataFrame
        Input dataframe from which numeric columns will be selected

    Returns
    -------
    list[dict[str, Any]]
        A list of dictionaries, one per numeric column, with the following keys:
        - "column" (str): column name
        - "mean" (float): mean, rounded to 2 decimal places
        - "median" (float): median, rounded to 2 decimal places
        - "mode" (list[float]): up to top 5 mode values as a list
        - "std" (float): standard deviation, rounded to 2 decimal places
        - "variance" (float): variance, rounded to 2 decimal places
        - "range" (float): max - min, rounded to 2 decimal places
        - "min" (numeric): minimum value (no additional rounding applied)
        - "max" (numeric): maximum value (no additional rounding applied)
        - "count" (int): number of non-missing observations
        - "missing_values" (int): number of missing entries
        - "missing_pct" (float): fraction of missing entries (rounded to 2 decimals)
        - "unique_values" (int): number of unique values

    Raises
    ------
    None
        This function handles empty series by printing the global EMPTY_DATAFRAME
        constant and skipping the column; it does not raise on empty columns.
        Any unexpected exceptions from underlying series methods will propagate.

    Examples
    --------
    >>> # Returns a list of dicts with summary stats for each numeric column
    >>> summaries = get_numeric_summary_stats(my_dataframe)
    """
    numeric_summary_stats: list[dict[str, Any]] = []

    for col in data.select(n_cs.numeric()).columns:
        series = data[col]

        if len(series) == 0:
            print(EMPTY_DATAFRAME)
            continue

        # Central tendency: mean, median and mode
        mean: float = series.mean().__round__(2)
        median: float = series.median().__round__(2)
        mode: list[float] = series.mode().to_list()[:5]  # Top 5 modes

        # Spread: std, variance, range, iqr_value, min, max
        std: float = series.std().__round__(2)
        variance: float = series.var().__round__(2)
        data_range: float = (series.max() - series.min()).__round__(2)
        min_value: float = series.min()
        max_value: float = series.max()

        # Others: count, missing_values, unique_values
        count: int = series.count()
        missing_values: int = series.is_null().sum()  # type: ignore
        missing_pct: float = (missing_values / series.shape[0]).__round__(2)
        unique_values: int = series.n_unique()

        numeric_summary_stats.append(
            {
                "column": col,
                "mean": mean,
                "median": median,
                "mode": mode,
                "std": std,
                "variance": variance,
                "range": data_range,
                "min": min_value,
                "max": max_value,
                "count": count,
                "missing_values": missing_values,
                "missing_pct": missing_pct,
                "unique_values": unique_values,
            }
        )

    return numeric_summary_stats


def get_categorical_summary_stats(data: nw.DataFrame) -> list[dict[str, Any]]:
    """
    Compute summary statistics for categorical (string) columns in a DataFrame.

    Parameters
    ----------
    data : nw.DataFrame
        DataFrame-like object that exposes a `.select(...)` method and column-access semantics.
        Only columns selected by `n_cs.string()` (string/categorical columns) will be processed.

    Returns
    -------
    list[dict[str, Any]]
        A list of dictionaries, one per categorical column. Each dictionary contains the keys:
        - column (str): Column name.
        - total_count (int): Number of non-missing (observed) entries in the column.
        - unique_values (int): Number of distinct values in the column.
        - value_counts (list[list[Any]]): Frequency counts as a list of [value, count] pairs
          (converted from the underlying value_counts result and returned as plain Python lists).
        - missing_values (int): Number of missing entries in the column.
        - missing_pct (float): Percentage of missing entries relative to the column length,
          rounded to 2 decimal places.

    Raises
    ------
    AttributeError, TypeError
        May raise if `data` does not implement the expected DataFrame-like API (e.g., missing
        `.select`, column access, or aggregation methods used internally).

    Examples
    --------
    >>> # Returns a list of summary dicts for each categorical column
    >>> summaries = get_categorical_summary_stats(df)
    >>> summaries[0]['value_counts']
    [['red', 10], ['blue', 7], ['green', 3]]
    """
    summary_stats: list[dict[str, Any]] = []

    for col in data.select(n_cs.string()).columns:
        series = data[col]

        if len(series) == 0:
            print(EMPTY_DATAFRAME)
            continue

        # Frequency counts and percentages
        value_counts: list[list[Any]] = (
            series.value_counts(sort=True)
            .to_numpy()
            .tolist()[:10]  # Top 10 unique values
        )

        # Basic stats: count, missing_values, missing_pct, unique_values
        count: int = series.count()
        missing_values: int = series.is_null().sum()  # type: ignore
        missing_pct: float = (missing_values / series.shape[0] * 100).__round__(2)
        unique_values: int = series.n_unique()

        summary_stats.append(
            {
                "column": col,
                "total_count": count,
                "unique_values": unique_values,
                "value_counts": value_counts,
                "missing_values": missing_values,
                "missing_pct": missing_pct,
            }
        )
    return summary_stats


def to_nw_df(data: IntoFrameT) -> nw.DataFrame:
    """
    Convert a Pandas/Polars/PyArrow Table into an nw.DataFrame.

    Parameters
    ----------
    data : IntoFrameT
        Input data to convert. Supported input types are those accepted by
        nw.from_native (for example, pandas.DataFrame, dict, list of dicts,
        numpy.ndarray, pyarrow.Table, or other native tabular representations).

    Returns
    -------
    nw.DataFrame
        An nw.DataFrame instance representing the converted data.

    Raises
    ------
    TypeError
        If the provided `data` type is not supported by nw.from_native.
    ValueError
        If the input contains invalid or inconsistent contents that prevent conversion.

    Notes
    -----
    This function is a thin wrapper around nw.from_native and preserves its
    conversion semantics and error behavior.

    Examples
    --------
    >>> to_nw_df(my_pandas_df)
    <nw.DataFrame ...>
    """
    return nw.from_native(data)


def get_numeric_schema(data: nw.DataFrame) -> dict[str, Any]:
    """
    Construct a mapping of numeric column names to their schema types represented as strings.

    Parameters
    ----------
    data : nw.DataFrame
        A dataframe-like object that supports selection of numeric columns via
        `select(n_cs.numeric())` and exposes a `collect_schema()` method which
        returns a mapping of column names to schema/type objects.

    Returns
    -------
    dict[str, str]
        A dictionary where keys are numeric column names from `data` and values
        are the string representation of each column's schema/type (obtained
        by applying `str()` to the collected schema values).

    Raises
    ------
    AttributeError
        If `data` does not implement the expected interface (e.g., missing
        `select` or `collect_schema`), an AttributeError may be raised by the
        attempted method calls.
    Exception
        Any exceptions raised by the underlying `collect_schema()` call are
        propagated.

    Examples
    --------
    >>> # Assuming `df` is an nw.DataFrame with numeric columns:
    >>> get_numeric_schema(df)
    {'age': 'IntegerType', 'salary': 'FloatType'}
    """
    numeric_schema: dict[str, Any] = {
        k: str(v) for k, v in data.select(n_cs.numeric()).collect_schema().items()
    }
    return numeric_schema


def get_string_schema(data: nw.DataFrame) -> dict[str, Any]:
    """
    Create a schema mapping for string-typed columns in a DataFrame.

    Parameters
    ----------
    data : nw.DataFrame
        A DataFrame object

    Returns
    -------
    dict[str, Any]
        A dictionary where keys are column names for string-typed columns and values are
        the stringified schema/type value as returned by collect_schema().

    Raises
    ------
    TypeError
        If `data` does not provide the expected methods (e.g. select or collect_schema).

    Examples
    --------
    >>> get_string_schema(df)
    {'first_name': 'StringType', 'city': 'StringType'}
    """
    string_schema: dict[str, Any] = {
        k: str(v) for k, v in data.select(n_cs.string()).collect_schema().items()
    }
    return string_schema


def get_cardinality_info(data: nw.DataFrame) -> dict[str, Any]:
    """
    Compute the cardinality (number of unique values) for numeric and string columns.

    Parameters
    ----------
    data : nw.DataFrame
        A dataframe-like object from the `nw` library.

    Returns
    -------
    dict[str, Any]
        A dictionary containing two entries:
        - "num_unique_numeric_rows": dict
            A mapping from numeric column names to the number of unique values
            present in each numeric column.
        - "num_unique_categorical_rows": dict
            A mapping from string column names to the number of unique values
            present in each string column.

    Raises
    ------
    TypeError
        If `data` is not a compatible `nw.DataFrame` or does not provide the
        expected selection/column APIs.
    ValueError
        If no numeric or string columns can be identified (optional, depending
        on the underlying dataframe implementation).

    Examples
    --------
    >>> # Assuming `df` is an nw.DataFrame with numeric column "age" and string
    >>> # column "city":
    >>> get_cardinality_info(df)
    {
        "num_unique_numeric_rows": {"age": 42},
        "num_unique_categorical_rows": {"city": 12}
    """
    cardinality: dict[str, int] = {  # type: ignore
        "num_unique_numeric_rows": {
            col: data[col].n_unique()
            for col in data.select(n_cs.numeric()).columns  # type: ignore
        },
        "num_unique_categorical_rows": {
            col: data[col].n_unique()
            for col in data.select(n_cs.string()).columns  # type: ignore
        },
    }
    return cardinality


def get_null_info(data: nw.DataFrame) -> dict[str, Any]:
    """
    Return information about null values in a DataFrame.

    Parameters
    ----------
    data : nw.DataFrame
        A DataFrame-like object that implements a null_count() method.
    Returns
    -------
    dict[str, Any]
        Dictionary containing:
        - "data_nulls" (dict[str, int]): Mapping from column names to the
          number of null values in each column.
        - "total_nulls" (int): Total number of null values across the entire
          DataFrame.

    Raises
    ------
    AttributeError
        If the provided object does not implement the expected methods
        (e.g., null_count, to_polars, to_numpy) or their return values do not
        support the conversions used.

    Example
    -------
    >>> get_null_info(df)
    {'data_nulls': {'col1': 0, 'col2': 3}, 'total_nulls': 3}
    """
    null_info: dict[str, Any] = {
        "data_nulls": data.null_count().to_polars().to_dicts()[0],
        "total_nulls": data.null_count().to_numpy().sum().item(),
    }
    return null_info


def get_duplicated_rows_info(data: nw.DataFrame) -> dict[str, Any]:
    """
    Return information about duplicated rows in the given DataFrame.

    Parameters
    ----------
    data : nw.DataFrame
        The input DataFrame to inspect for duplicated rows. This function relies
        on the DataFrame's is_duplicated() method to identify duplicates.

    Returns
    -------
    dict[str, Any]
        A dictionary containing at least the following key:
        - "num_duplicated_rows" (int): The number of rows marked as duplicates by
          data.is_duplicated(). This count reflects duplicate occurrences only,
          i.e. rows considered duplicates after the first occurrence in each
          duplicated group.

    Examples
    --------
    >>> # Assuming `df` is an nw.DataFrame
    >>> # result = get_duplicated_rows_info(df)
    >>> # result["num_duplicated_rows"]
    """
    return {"num_duplicated_rows": data.is_duplicated().sum()}


def get_memory_usage_info(data: nw.DataFrame) -> dict[str, Any]:
    """
    Get memory usage information for a DataFrame-like object.

    Parameters
    ----------
    data : nw.DataFrame
        The DataFrame object to analyze.

    Returns
    -------
    dict[str, Any]
        A dictionary containing:
        - memory_usage_MB (float): Estimated memory usage in megabytes, rounded to
          two decimal places.
        - validation_timestamp (str): ISO 8601 timestamp (seconds precision)
          indicating when the measurement was taken.

    Raises
    ------
    AttributeError
        If the provided `data` object does not have an `estimated_size` attribute.
    ValueError
        If `estimated_size` returns a non-numeric value that cannot be converted to float.

    Notes
    -----
    This function calls `data.estimated_size(unit="mb")` to obtain the size and uses
    `pendulum.now().isoformat(timespec="seconds")` to produce the timestamp.

    Examples
    --------
    >>> get_memory_usage_info(df)
    {'memory_usage_MB': 12.34, 'validation_timestamp': '2025-09-27T12:34:56'}
    """
    return {
        "memory_usage_MB": round(data.estimated_size(unit="mb"), 2),
        "validation_timestamp": pendulum.now().isoformat(timespec="seconds"),
    }


def _data_validator(input_data: DataValidatorConfig) -> DataValidatorSchema:
    """
    Validate and summarize a dataset according to the provided DataValidatorConfig.

    Parameters
    ----------
    input_data : DataValidatorConfig
        Configuration object that drives validation. Expected attributes:
        - data: The raw dataset to be validated. It will be converted to the
          internal "nw" DataFrame via `to_nw_df`.
        - schema_fns: Iterable[Callable[[DataFrame], dict]] - functions that accept
          the converted DataFrame and return schema descriptions. If two schema
          functions are provided, the first is expected to describe numeric
          columns and the second string/categorical columns.
        - info_fns: Iterable[Callable[[DataFrame], dict]] - functions that accept
          the converted DataFrame and return additional information; returned
          dictionaries will be merged.
        - summary_fns: Iterable[Callable[[DataFrame], dict]] - functions that accept
          the converted DataFrame and return summary statistics. If two summary
          functions are provided, the first is expected to be numeric summaries
          and the second categorical summaries.

    Returns
    -------
    DataValidatorSchema
        A structured schema object (Pydantic model) containing the validation results.
        The top-level keys in the returned object typically include:
        - data_schema: dict - schema information for numeric and string columns

    Raises
    ------
    Any exception raised by the underlying helpers (e.g. `to_nw_df`, the schema,
    info or summary functions, or column selectors) will propagate. No special
    error translation is performed.

    Examples
    --------
    Assuming `cfg` is an instance of DataValidatorConfig:
    >>> result = _data_validator(cfg)
    >>> result["data_shape"]["total_rows"]
    >>> result["summary_statistics"]["numeric"]
    """
    nw_data = to_nw_df(input_data.data)

    # Collect schema information
    schema: list[dict[str, Any]] = [
        schema_fn(nw_data) for schema_fn in input_data.schema_fns
    ]

    # Collect general info
    info: dict[str, Any] = {}
    for info_fn in input_data.info_fns:
        info.update(info_fn(nw_data))

    # Collect summary statistics
    summary_stats: list[dict[str, Any]] = [
        stats_fn(nw_data)  # type: ignore
        for stats_fn in input_data.summary_fns
    ]

    result: dict[str, Any] = {
        "data_schema": (
            {"numeric": schema[0], "string": schema[1]} if len(schema) == 2 else schema
        ),
        "data_shape": {
            "total_rows": nw_data.shape[0],
            "total_columns": nw_data.shape[1],
            "number_of_numeric_columns": len(
                list(nw_data.select(n_cs.numeric()).columns)
            ),
            "number_of_categorical_columns": len(
                list(nw_data.select(n_cs.string()).columns)
            ),
        },
        "summary_statistics": {
            "numeric": summary_stats[0] if len(summary_stats) == 2 else [],
            "categorical": summary_stats[1] if len(summary_stats) == 2 else [],
        },
        "other_info": info,
    }
    return DataValidatorSchema(**result)  # type: ignore


def data_validator(data: IntoFrameT) -> DataValidatorSchema:
    """
    Validate and summarize tabular data using predefined schema, info, and summary functions.

    Parameters
    ----------
    data : IntoFrameT
        Input dataset to validate. Accepted types are Pandas/Polars DataFrame,
        or PyArrow Table.

    Returns
    -------
    DataValidatorSchema
        A Pydantic model instance containing the validation results, including:
        - data_schema: dict with numeric and categorical schema information
        - data_shape: dict with total rows/columns and counts of numeric/categorical columns
        - summary_statistics: dict with lists of numeric and categorical summary stats
        - other_info: dict with additional metadata (e.g., null counts, memory usage)

    Raises
    ------
    TypeError
        If `data` is not a supported input type and cannot be converted to the expected
        frame representation.
    ValueError
        If required schema/info/summary functions fail to produce valid outputs.

    Examples
    --------
    >>> # Validate a Polars DataFrame
    >>> result = data_validator(df)
    >>> isinstance(result, dict)
    True
    """
    try:
        config = DataValidatorConfig(
            data=data,
            schema_fns=[get_numeric_schema, get_string_schema],
            info_fns=[
                get_cardinality_info,
                get_null_info,
                get_duplicated_rows_info,
                get_memory_usage_info,
            ],
            summary_fns=[
                get_numeric_summary_stats,
                get_categorical_summary_stats,
            ],
        )
        return _data_validator(config)

    except DataValidationError as e:
        raise e
