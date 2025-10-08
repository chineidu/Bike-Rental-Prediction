from enum import Enum


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
