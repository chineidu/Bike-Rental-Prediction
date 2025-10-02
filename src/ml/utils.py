from typing import Any

import narwhals as nw
import numpy as np
import polars as pl
from narwhals.typing import IntoDataFrameT
from sklearn.base import BaseEstimator
from sklearn.metrics._regression import mean_absolute_error, root_mean_squared_error
from sklearn.model_selection import TimeSeriesSplit


def split_temporal_data(
    data: IntoDataFrameT, test_size: float = 0.2
) -> tuple[IntoDataFrameT, IntoDataFrameT]:
    """Split data into training and testing sets while maintaining temporal order."""

    nw_data: nw.DataFrame = nw.from_native(data)
    train_size: float = int((1 - test_size) * nw_data.shape[0])

    nw_data = nw_data.with_row_index()
    train_data: nw.DataFrame = nw_data.filter(nw.col("index") < train_size).drop(
        "index"
    )
    test_data: nw.DataFrame = nw_data.filter(nw.col("index") >= train_size).drop(
        "index"
    )

    return (train_data.to_native(), test_data.to_native())


def split_into_train_test(
    train_df: IntoDataFrameT, test_df: IntoDataFrameT, target_col: str
) -> dict[str, Any]:
    """Split data into features and target for training and testing sets."""
    x_train = train_df.drop(target_col).to_numpy()
    y_train = train_df[target_col].to_numpy()

    x_test = test_df.drop(target_col).to_numpy()
    y_test = test_df[target_col].to_numpy()

    columns = train_df.drop(target_col).columns

    return {
        "x_train": x_train,
        "y_train": y_train,
        "x_test": x_test,
        "y_test": y_test,
        "columns": columns,
    }


def compute_metrics(
    y_true: np.ndarray | list[float | int], y_pred: np.ndarray | list[float | int]
) -> dict[str, float]:
    """
    Compute evaluation metrics between true and predicted values.

    Metrics returned:
    - MAPE: Mean Absolute Percentage Error (in %)
    - MAE: Mean Absolute Error
    - RMSE: Root Mean Squared Error

    Parameters:
    ----------
    y_true : array-like
        Ground truth values.
    y_pred : array-like
        Predicted values.

    Returns:
    -------
    dict
        Dictionary with keys 'MAPE', 'MAE', and 'RMSE' and their float values.
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = root_mean_squared_error(y_true, y_pred)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    mape: float = (
        np.mean(np.abs((y_true - y_pred) / np.where(y_true == 0, 0.01, y_true))) * 100
    ).item()

    return {
        "MAE": round(mae, 2),
        "RMSE": round(rmse, 2),
        "MAPE": round(mape, 2),
    }


def _calculate_corr(x: list[float] | np.ndarray, y: list[float] | np.ndarray) -> float:
    return np.corrcoef(x, y)[0][1].item()


def compute_autocorrelation(series: nw.Series, max_lag: int = 24) -> dict[int, float]:
    """
    Compute autocorrelation for a time series using Polars.

    Parameters
    ----------
    series : pl.Series
        The time series data.
    max_lag : int, default=24
        Maximum lag to compute autocorrelation for.

    Returns
    -------
    dict[int, float]
        Dictionary mapping lag to autocorrelation value.
    """
    autocorr_values: dict[int, float] = {}

    for lag in range(1, max_lag + 1):
        try:
            # Create a DataFrame with original and lagged series
            df_corr: nw.DataFrame = nw.from_native(
                pl.DataFrame({"original": series, "lagged": series.shift(lag)})
            ).drop_nulls()

            # Compute correlation if we have sufficient data
            if df_corr.shape[0] > 1:
                correlation = _calculate_corr(df_corr["original"], df_corr["lagged"])
                autocorr_values[lag] = correlation
            else:
                autocorr_values[lag] = None  # type: ignore

        except Exception as e:
            print(f"Error computing lag {lag}: {e}")
            autocorr_values[lag] = None  # type: ignore

    return autocorr_values


def drop_features(data: nw.DataFrame, features: list[str]) -> nw.DataFrame:
    """Drop specified features from the dataset."""

    return data.drop(features)


def cross_validate_sklearn_model(
    tscv: TimeSeriesSplit,
    x_train: np.ndarray,
    y_train: np.ndarray,
    model: BaseEstimator,
    verbose: bool = False,
) -> dict[str, Any]:
    """Cross-validate a sklearn model using TimeSeriesSplit."""
    all_rmse: list[float] = []
    all_mae: list[float] = []
    all_mape: list[float] = []

    for i, (train_index, test_index) in enumerate(tscv.split(x_train), start=1):
        if verbose:
            print(f"Fold {i}:")
        x_tr, x_val = x_train[train_index], x_train[test_index]
        y_tr, y_val = y_train[train_index], y_train[test_index]

        # Train and evaluate the model
        model.fit(x_tr, y_tr)  # type: ignore
        y_pred = model.predict(x_val)  # type: ignore
        metrics = compute_metrics(y_val, y_pred)
        if verbose:
            print(f"Validation Metrics: {metrics}")

        all_rmse.append(metrics.get("RMSE"))  # type: ignore
        all_mae.append(metrics.get("MAE"))  # type: ignore
        all_mape.append(metrics.get("MAPE"))  # type: ignore

    return {"model": model, "RMSE": all_rmse, "MAE": all_mae, "MAPE": all_mape}
