from enum import Enum
from typing import Any, TypedDict

import numpy as np

type Number = int | float | np.integer | np.floating


# =======================================
# ============= ENUMS ===================
# =======================================
class ArtifactsType(str, Enum):
    """Enum for different types of artifacts."""

    JSON = "json"
    TXT = "txt"
    YAML = "yaml"
    PICKLE = "pkl"

    def __str__(self) -> str:
        return str(self.value)


class ModelType(str, Enum):
    """Enum for different types of models."""

    RANDOM_FOREST = "RandomForestRegressor"
    XGBOOST = "XGBoostRegressor"
    LIGHTGBM = "LightGBMRegressor"


class CurrencyType(str, Enum):
    """Enum for different currency types."""

    NGN = "NGN"
    USD = "USD"


class UserRole(str, Enum):
    """Enum for different user roles."""

    ADMIN = "admin"
    USER = "user"
    MODERATOR = "moderator"

    def __str__(self) -> str:
        return str(self.value)


# =======================================
# ============== DATA ===================
# =======================================
class DataDict(TypedDict):
    """Dictionary type for training, validation, and test data splits."""

    x_train: list[list[Number]] | np.ndarray
    y_train: list[Number] | np.ndarray
    x_val: list[list[Number]] | np.ndarray
    y_val: list[Number] | np.ndarray
    x_test: list[list[Number]] | np.ndarray
    y_test: list[Number] | np.ndarray


class MetricsDict(TypedDict):
    """Dictionary type for model evaluation metrics."""

    MAE: float
    RMSE: float
    MAPE: float
    Adjusted_R2: float | None


class WeatherDict(TypedDict):
    temp: float
    hum: float
    windspeed: float
    weathersit: float


class ModelManagerDict(TypedDict):
    model_loader: Any
    prediction_service: Any


class PredictionResultDict(TypedDict):
    base_price: float
    price: float
    competitor_price: float
    price_multiplier: float
    surge: float
    discount: float
    min_price: float
    max_price: float
    demand: float
    currency: str
