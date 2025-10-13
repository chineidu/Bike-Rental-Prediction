from typing import Any, Literal

import lightgbm as lgb
import narwhals as nw
import numpy as np
import polars as pl
import xgboost as xgb
from narwhals.typing import IntoDataFrameT
from sklearn.base import BaseEstimator
from sklearn.metrics import r2_score
from sklearn.metrics._regression import mean_absolute_error, root_mean_squared_error
from sklearn.model_selection import TimeSeriesSplit

from src.schemas.types import MetricsDict, Number


def split_temporal_data_to_train_val_test(
    data: IntoDataFrameT,
    test_size: float = 0.2,
    val_size: float = 0.2,
    print_shapes: bool = True,
) -> tuple[IntoDataFrameT, IntoDataFrameT, IntoDataFrameT]:
    """Split data into train, validation and test sets while maintaining temporal order.

    Parameters
    ----------
    data : IntoDataFrameT
        The input dataframe to be split.
    test_size : float, default=0.2
        Proportion of the dataset to include in the test split.
    val_size : float, default=0.2
        Proportion of the dataset to include in the validation split.
    print_shapes : bool, default=True
        Whether to print the shapes of the resulting dataframes.

    Returns
    -------
    tuple[IntoDataFrameT, IntoDataFrameT, IntoDataFrameT]
        A tuple containing the train, validation, and test dataframes.
    """
    # Validation
    if not 0.0 <= test_size <= 1.0 or not 0.0 <= val_size <= 1.0:
        raise ValueError("test_size and val_size must be between 0.0 and 1.0")
    if (test_size + val_size) >= 1.0:
        raise ValueError("The sum of test_size and val_size must be less than 1.0")

    # Convert to Narwhals DataFrame
    nw_data: nw.DataFrame = nw.from_native(data)
    num_train: int = int((1 - test_size - val_size) * nw_data.shape[0])
    num_val: int = int(val_size * nw_data.shape[0])

    # Add index
    nw_data = nw_data.with_row_index()
    train_data: nw.DataFrame = nw_data.filter(nw.col("index") <= num_train).drop(
        "index"
    )

    val_data: nw.DataFrame = nw_data.filter(
        (nw.col("index") > num_train) & (nw.col("index") <= (num_train + num_val))
    ).drop("index")
    test_data: nw.DataFrame = nw_data.filter(
        nw.col("index") > (num_train + num_val)
    ).drop("index")

    if print_shapes:
        print(
            f"Shapes -> Train shape: {train_data.shape} | Val shape: {val_data.shape} | Test shape: {test_data.shape}"
        )

    return (train_data.to_native(), val_data.to_native(), test_data.to_native())


def split_into_train_test_sets(
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


def adjusted_r2_score(y_true: np.ndarray, y_pred: np.ndarray, n_features: int) -> float:
    """
    Calculate the adjusted R-squared using sklearn's r2_score.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground truth (correct) target values.
    y_pred : array-like of shape (n_samples,)
        Estimated target values.
    n_features : int
        Number of independent variables (features) used in the model.

    Returns
    -------
    float
        Adjusted R-squared value.

    Notes
    -----
    The adjusted R-squared is computed as:

    1 - (1 - R^2) * (n - 1) / (n - p - 1)

    where n is the number of observations and p is the number of features.
    """
    n = len(y_true)
    r2 = r2_score(y_true, y_pred)
    return 1 - (1 - r2) * (n - 1) / (n - n_features - 1)


def compute_metrics(
    y_true: Number | np.ndarray,
    y_pred: Number | np.ndarray,
    n_features: int | None = None,
) -> MetricsDict:
    """
    Compute evaluation metrics between true and predicted values.

    Metrics returned:
    - MAPE: Mean Absolute Percentage Error (in %)
    - MAE: Mean Absolute Error
    - RMSE: Root Mean Squared Error
    - Adjusted R² (if n_features is provided)

    Parameters:
    ----------
    y_true : array-like
        Ground truth values.
    y_pred : array-like
        Predicted values.
    n_features : int, optional
        Number of features used in the model (required for Adjusted R²).

    Returns:
    -------
    MetricsDict
        Dictionary containing the computed metrics.
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = root_mean_squared_error(y_true, y_pred)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    mape: float = (
        np.mean(np.abs((y_true - y_pred) / np.where(y_true == 0, 0.01, y_true))) * 100
    ).item()

    if n_features:
        adj_r2 = adjusted_r2_score(y_true, y_pred, n_features)

    return MetricsDict(
        MAE=round(mae, 2),
        RMSE=round(rmse, 2),
        MAPE=round(mape, 2),
        Adjusted_R2=round(adj_r2, 2) if n_features else None,
    )


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
            df_corr: nw.DataFrame = nw.from_native(  # type: ignore
                pl.DataFrame({"original": series, "lagged": series.shift(lag)})
            ).drop_nulls()

            # Compute correlation if we have sufficient data
            if df_corr.shape[0] > 1:
                correlation = _calculate_corr(df_corr["original"], df_corr["lagged"])  # type: ignore
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
    all_adj_r2: list[float | None] = []

    for i, (train_index, test_index) in enumerate(tscv.split(x_train), start=1):
        if verbose:
            print(f"Fold {i}:")
        x_tr, x_val = x_train[train_index], x_train[test_index]
        y_tr, y_val = y_train[train_index], y_train[test_index]

        # Train and evaluate the model
        model.fit(x_tr, y_tr)  # type: ignore
        y_pred = model.predict(x_val)  # type: ignore
        metrics: MetricsDict = compute_metrics(
            y_val, y_pred, n_features=x_train.shape[1]
        )
        if verbose:
            print(f"Validation Metrics: {metrics}")

        all_rmse.append(metrics.get("RMSE"))  # type: ignore
        all_mae.append(metrics.get("MAE"))  # type: ignore
        all_mape.append(metrics.get("MAPE"))  # type: ignore
        all_adj_r2.append(metrics.get("Adjusted_R2"))  # type: ignore

    return {
        "trained_model": model,
        "metrics": {
            "RMSE": np.mean(all_rmse).round(2).item(),
            "MAE": np.mean(all_mae).round(2).item(),
            "MAPE": np.mean(all_mape).round(2).item(),
            "Adjusted_R2": np.mean(all_adj_r2).round(2).item(),
        },
    }


def get_model_feature_importance(
    model_name: str, features: list[str], weights: np.ndarray | list[float], n: int = 20
) -> tuple[pl.DataFrame, dict[str, float]]:
    """
    Compute the top feature importances for a trained model.

    Parameters
    ----------
    model_name : str
        Name of the model used for tagging the feature importances.
    features : list[str]
        Ordered list of feature names corresponding to `weights`.
    weights : np.ndarray | list[float]
        Importance scores aligned with `features`.
    n : int, default=20
        Number of top features to retain.

    Returns
    -------
    tuple[pl.DataFrame, dict[str, float]]
        A tuple containing the Polars DataFrame of top features and a dictionary of
        tags formatted for logging.
    """
    feature_importance = (
        pl.DataFrame({"feature": features, "importance": weights})
        .sort(by="importance", descending=True)
        .head(n)
    )
    tags: dict[str, Any] = {
        f"{model_name}_top_feature_{idx}": f"{row[0]} ({row[1]:.4f})"
        for idx, row in enumerate(feature_importance.iter_rows(), start=1)
    }
    return (feature_importance, tags)


def get_feature_importance_from_booster(
    model: xgb.Booster,
    feature_names: list[str],
    importance_type: Literal[
        "weight", "gain", "cover", "total_gain", "total_cover"
    ] = "gain",
) -> dict[str, float]:
    """
    Extract feature importance from XGBoost Booster object.

    Parameters
    ----------
    model : xgb.Booster
        Trained XGBoost Booster model.
    feature_names : list[str]
        List of feature names corresponding to training data columns.
    importance_type : str, default="weight"
        Type of importance to extract. Options:
        - "weight": Number of times a feature appears in trees
        - "gain": Average gain across all splits using the feature
        - "cover": Average coverage across all splits using the feature
        - "total_gain": Total gain across all splits using the feature
        - "total_cover": Total coverage across all splits using the feature

    Returns
    -------
    dict[str, float]
        Dictionary mapping feature names to importance scores.
    """
    # Get feature importance scores from booster
    importance_dict = model.get_score(importance_type=importance_type)
    # Normalize importance scores
    sum_: float = sum(importance_dict.values())
    imp_dict = {feat: round(imp / sum_, 5) for feat, imp in importance_dict.items()}  # type: ignore
    # Map feature indices (f0, f1, f2...) to actual feature names
    feature_importance: dict[str, float] = {
        feature_name: imp_dict.get(f"f{idx}", 0.0)
        for idx, feature_name in enumerate(feature_names)
    }

    return feature_importance


def get_lightgbm_feature_importance(
    model: lgb.Booster, features: list[str]
) -> pl.DataFrame:
    """
    Get feature importance from a LightGBM model.

    Parameters
    ----------
    model : lgb.Booster
        Trained LightGBM model
    features : list[str]
        List of feature names

    Returns
    -------
    pl.DataFrame
        DataFrame containing feature importance scores
    """

    # Get different types of importance
    split_importance = model.feature_importance(importance_type="split")
    gain_importance = model.feature_importance(importance_type="gain")

    # Create DataFrame for easy analysis
    importance_df = pl.DataFrame(
        {
            "feature": features,
            "split_importance": split_importance,
            "gain_importance": gain_importance,
        }
    )

    # Normalize importance scores
    return importance_df.with_columns(
        (pl.col("split_importance") / pl.col("split_importance").sum()).alias(
            "split_importance_normalized"
        ),
        (pl.col("gain_importance") / pl.col("gain_importance").sum()).alias(
            "gain_importance_normalized"
        ),
    )


def extract_metrics(data: dict[str, Any]) -> dict[str, Any]:
    """
    Extract and flatten metrics from a mapping into a single dictionary.

    Parameters
    ----------
    data : dict[str, Any]
        Input mapping that may contain a "metrics" key whose value is itself a
        mapping of metric names to metric values. Other keys are copied through
        to the result as-is.

    Returns
    -------
    dict[str, Any]
    """
    results = {}
    for key, val in data.items():
        if key == "metrics":
            for metric_name, metric_value in val.items():
                results[metric_name] = metric_value  # noqa: PERF403
        else:
            results[key] = val
    return results


def create_metrics_df(data_list: list[dict[str, Any]]) -> pl.DataFrame:
    """
    Create a DataFrame of metrics extracted from a sequence of input records.

    Parameters
    ----------
    data_list : list[dict[str, Any]]
        A list of input records. Each record should be a mapping containing the raw
        data expected by `extract_metrics`.

    Returns
    -------
    pl.DataFrame
    """
    all_metrics: list[dict[str, Any]] = [extract_metrics(data) for data in data_list]
    return pl.DataFrame(all_metrics)


def combine_train_val(
    x_train: np.ndarray, y_train: np.ndarray, x_val: np.ndarray, y_val: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Combine training and validation feature/target arrays into a single train set.

    Parameters
    ----------
    x_train : np.ndarray
        Training features, shape (n_train, n_features).
    y_train : np.ndarray
        Training targets, shape (n_train,) or (n_train, 1).
    x_val : np.ndarray
        Validation features, shape (n_val, n_features).
    y_val : np.ndarray
        Validation targets, shape (n_val,) or (n_val, 1).

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Tuple containing:
        - X : np.ndarray, stacked features with shape (n_train + n_val, n_features)
        - y : np.ndarray, 1D stacked targets with shape (n_train + n_val,)

    Raises
    ------
    ValueError
        If input dimensions are incompatible (mismatched feature counts or lengths).
    """
    # Validation
    if x_train.ndim != 2 or x_val.ndim != 2:
        raise ValueError("x_train and x_val must be 2D arrays (n_samples, n_features).")
    if x_train.shape[1] != x_val.shape[1]:
        raise ValueError("Feature dimension mismatch between x_train and x_val.")
    if x_train.shape[0] != y_train.ravel().shape[0]:
        raise ValueError("Number of rows in x_train and y_train must match.")
    if x_val.shape[0] != y_val.ravel().shape[0]:
        raise ValueError("Number of rows in x_val and y_val must match.")

    # Stack features and concatenate targets into a 1D array
    X = np.vstack((x_train, x_val))
    y = np.vstack((y_train.reshape(-1, 1), y_val.reshape(-1, 1))).ravel()

    return X, y
