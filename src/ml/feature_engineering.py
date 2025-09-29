from dataclasses import dataclass
from typing import Literal

import narwhals as nw
import numpy as np
import polars as pl
from narwhals.typing import IntoDataFrameT

from .utils import drop_features


@dataclass
class Lags:
    feature: str
    lags: list[int]


@dataclass
class InteractionFeats:
    feature_1: str
    feature_2: str
    operation: Literal["add", "multiply"]


@dataclass
class Windows:
    feature: str
    windows: list[int]


@dataclass
class FeatureConfig:
    lags: list[Lags]
    diffs: list[Lags]
    interactions: list[InteractionFeats]
    rolling_windows: list[Windows]
    drop_feats: list[str]
    target_col: str


class FeatureEngineer:
    """
    Class for applying a configurable feature engineering pipeline to tabular data.
    The pipeline converts the input into a neutral DataFrame representation, generates
    a set of commonly useful features (temporal, cyclical, lag, rolling, interaction,
    difference, and binary features), drops configured columns, and returns the result
    in the native input format.

    Parameters
    ----------
    data : IntoDataFrameT
        Input data in a type that can be converted to the neutral dataframe used
        internally (e.g., pandas.DataFrame, numpy array, or other supported types).
    config : FeatureConfig
        Configuration object that controls which features to create. Expected fields
        include (but are not limited to): lags, rolling_windows, interactions,
        diffs, and drop_feats. Each entry should specify the target column(s) and
        parameters for the transformation.

    Attributes
    ----------
    data : nw.DataFrame
        Internal neutral dataframe representation used for feature operations.
    config : FeatureConfig
        Stored configuration used by the pipeline.

    Methods
    -------
        _create_all_features
        __repr__
        create_all_features
    """

    def __init__(self, data: IntoDataFrameT, config: FeatureConfig) -> None:
        self.data: nw.DataFrame = nw.from_native(data)
        self.config = config

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(data_shape={self.data.shape}, config={self.config})"

    def _create_all_features(self) -> nw.DataFrame:
        """Create and return a dataframe with all engineered features."""
        data = create_temporal_features(self.data)
        data = create_cyclical_features(data)

        # Lags
        for lag in self.config.lags:
            data = create_lag_features(data, target_col=lag.feature, lags=lag.lags)
        # Rolling
        for window in self.config.rolling_windows:
            data = create_rolling_features(
                data, target_col=window.feature, windows=window.windows
            )
        # Interactions
        for interaction in self.config.interactions:
            data = create_interaction_features(
                data,
                interaction.feature_1,
                interaction.feature_2,
                interaction.operation,
            )
        # Differences
        for diff in self.config.diffs:
            data = create_difference_features(
                data, target_col=diff.feature, lags=diff.lags
            )
        # Binary features
        data = create_binary_features(data)
        # Target variable
        data = create_target_variable(data, target_col=self.config.target_col)
        # Fill missing values
        data = fill_nulls(data, strategy="backward")
        # Drop features
        return drop_features(data, self.config.drop_feats)

    def create_all_features(self) -> IntoDataFrameT:
        """
        Create and return a dataframe with all engineered features.

        Parameters
        ----------
        self
            The instance on which the method is called. Expected to provide an
            implementation of `_create_all_features()`.

        Returns
        -------
        IntoDataFrameT
            A native dataframe-like object (the result of `self.data.to_native()`)
            containing the original input enhanced with all engineered features.

        Raises
        ------
        AttributeError
            If the instance does not implement `_create_all_features()`.
        TypeError
            If the object returned by `_create_all_features()` does not provide a
            `to_native()` method or if `to_native()` returns an unexpected type.
        ValueError
            If feature construction fails due to invalid or inconsistent input data.
        """
        self.data = self._create_all_features()
        return self.data.to_native()


# === Feature creation functions === #
def create_temporal_features(data: nw.DataFrame) -> nw.DataFrame:
    """Create temporal features."""
    weekend_list: list[int] = [5, 6]

    return data.with_columns(
        (nw.col("weekday").is_in(weekend_list)).cast(nw.Int8).alias("is_weekend")
    )


def create_cyclical_features(data: nw.DataFrame) -> nw.DataFrame:
    """Create cyclical features for hour and weekday."""
    df_pl: pl.DataFrame = data.to_polars().with_columns(
        # Hour cyclical features (24-hour cycle)
        (pl.col("hr") * (2 * np.pi / 24)).sin().alias("sin_hour"),
        (pl.col("hr") * (2 * np.pi / 24)).cos().alias("cos_hour"),
        # Weekday cyclical features (7-day cycle)
        (pl.col("weekday") * (2 * np.pi / 7)).sin().alias("sin_weekday"),
        (pl.col("weekday") * (2 * np.pi / 7)).cos().alias("cos_weekday"),
    )
    return nw.from_native(df_pl)  # type: ignore


def create_difference_features(
    data: nw.DataFrame, target_col: str, lags: list[int]
) -> nw.DataFrame:
    """Create difference features."""
    data_pl: pl.DataFrame = data.to_polars()
    for lag in lags:
        data_pl = data_pl.with_columns(
            pl.col(target_col).diff(lag).alias(f"{target_col}_diff_{lag}hr")
        )
    return nw.from_native(data_pl)  # type: ignore


def create_lag_features(
    data: nw.DataFrame, target_col: str, lags: list[int]
) -> nw.DataFrame:
    """Create lag features."""
    for lag in lags:
        data = data.with_columns(
            nw.col(target_col).shift(lag).alias(f"{target_col}_lag_{lag}hr")
        )
    return data


def create_interaction_features(
    data: nw.DataFrame,
    feature_1: str,
    feature_2: str,
    operation: Literal["add", "multiply"],
) -> nw.DataFrame:
    """Create interaction features."""
    if operation == "add":
        data = data.with_columns(
            (nw.col(feature_1) + nw.col(feature_2)).alias(
                f"{feature_1}_plus_{feature_2}"
            )
        )
    elif operation == "multiply":
        data = data.with_columns(
            (nw.col(feature_1) * nw.col(feature_2)).alias(
                f"{feature_1}_times_{feature_2}"
            )
        )
    return data


def create_rolling_features(
    data: nw.DataFrame, target_col: str, windows: list[int]
) -> nw.DataFrame:
    """Create rolling mean and median features."""
    data_pl: pl.DataFrame = data.to_polars()
    for window in windows:
        data_pl = data_pl.with_columns(
            pl.col(target_col)
            .rolling_mean(window)
            .alias(f"{target_col}_rolling_mean_{window}hr"),
            pl.col(target_col)
            .rolling_median(window)
            .alias(f"{target_col}_rolling_median_{window}hr"),
        )
    return nw.from_native(data_pl)  # type: ignore


def create_binary_features(data: nw.DataFrame) -> nw.DataFrame:
    """Create binary features based on thresholds and time of day."""
    high_percentile_temp: float = np.percentile(data["temp"], 80).item()
    high_percentile_hum: float = np.percentile(data["hum"], 80).item()
    working_hours: list[int] = [9, 10, 11, 12, 13, 14, 15, 16, 17]
    business_hours: list[int] = [8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
    peak_hours: list[int] = [7, 8, 9, 16, 17, 18]

    return data.with_columns(
        [
            # Binary features
            (nw.col("temp") > high_percentile_temp).cast(nw.Int8).alias("is_high_temp"),
            (nw.col("hum") > high_percentile_hum).cast(nw.Int8).alias("is_high_hum"),
            (nw.col("hr").is_in(peak_hours)).cast(nw.Int8).alias("is_peak_hour"),
            (nw.col("hr").is_in(working_hours)).cast(nw.Int8).alias("is_working_hour"),
            (nw.col("hr").is_in(business_hours))
            .cast(nw.Int8)
            .alias("is_business_hour"),
        ]
    )


def fill_nulls(
    data: nw.DataFrame,
    strategy: Literal[None, "forward", "backward", "min", "max", "mean", "zero", "one"],
) -> nw.DataFrame:
    """Fill missing values using the specified strategy."""
    df_pl = data.to_polars().fill_null(strategy=strategy)
    return nw.from_native(df_pl)  # type: ignore


def create_target_variable(data: nw.DataFrame, target_col: str) -> nw.DataFrame:
    """Create target variable by shifting the target column."""
    var_name: str = "target"
    """Shift the target column by -1 to predict the next time step."""
    return data.with_columns(nw.col(target_col).shift(-1).alias(var_name)).with_columns(
        nw.col(var_name).fill_null(strategy="forward")
    )
