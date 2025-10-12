from __future__ import annotations

from pathlib import Path
from typing import Literal

from omegaconf import DictConfig, OmegaConf
from pydantic import Field

from src import PACKAGE_PATH
from src.schemas import BaseSchema


class Lags(BaseSchema):
    feature: str = Field(
        ..., description="Feature name for which lags are to be created"
    )
    lags: list[int] = Field(..., description="List of lag periods")


class Diffs(BaseSchema):
    feature: str = Field(
        ..., description="Feature name for which diffs are to be created"
    )
    diffs: list[int] = Field(..., description="List of diff periods")


class InteractionFeats(BaseSchema):
    feature_1: str = Field(..., description="First feature for interaction")
    feature_2: str = Field(..., description="Second feature for interaction")
    operation: Literal["add", "multiply"] = Field(
        ..., description="Operation to perform on features"
    )


class Windows(BaseSchema):
    feature: str = Field(
        ..., description="Feature name for which windows are to be created"
    )
    windows: list[int] = Field(..., description="List of window periods")


class FeatureConfig(BaseSchema):
    """Feature engineering configuration class."""

    lag_features: list[Lags] = Field(..., description="List of lag configurations")
    diff_features: list[Diffs] = Field(..., description="List of diff configurations")
    interaction_features: list[InteractionFeats] = Field(
        ..., description="List of interaction configurations"
    )
    rolling_features: list[Windows] = Field(
        ..., description="List of rolling window configurations"
    )
    drop_features: list[str] = Field(..., description="List of features to drop")
    target_col: str = Field(..., description="Target column name")


class ExperimentTags(BaseSchema):
    project: str = Field(..., description="Project name")
    metric_of_interest: Literal["RMSE", "MAE", "MAPE", "MSE"] = Field(
        ..., description="Metric of interest"
    )
    others: dict[str, int | str] | None = Field(
        None, description="Additional tags as key-value pairs"
    )


class ExperimentConfig(BaseSchema):
    experiment_name: str = Field(..., description="Name of the experiment")
    experiment_tags: ExperimentTags = Field(..., description="Tags for the experiment")


class GeneralConfig(BaseSchema):
    random_seed: int = Field(..., description="Random seed for reproducibility")
    test_size: float = Field(
        ..., ge=0, le=1, description="Proportion of data to be used as test set"
    )
    cv_test_size: int = Field(
        ..., description="Size of the test set for cross-validation"
    )
    n_splits: int = Field(
        ..., description="Number of splits for time series cross-validation"
    )


class RandomForestConfig(BaseSchema):
    n_estimators: int = Field(..., description="Number of trees in the forest")
    max_depth: int | None = Field(None, description="Maximum depth of the tree")
    min_samples_split: int = Field(
        ..., description="Minimum number of samples required to split an internal node"
    )
    min_samples_leaf: int = Field(
        ..., description="Minimum number of samples required to be at a leaf node"
    )


class RandomForestOptunaConfig(BaseSchema):
    n_estimators: tuple[int, int] = Field(
        ..., description="Range for number of estimators"
    )
    max_depth: tuple[int, int] = Field(
        ..., description="Range for maximum depth of the tree"
    )
    min_samples_split: tuple[int, int] = Field(
        ..., description="Range for minimum samples to split an internal node"
    )
    min_samples_leaf: tuple[int, int] = Field(
        ..., description="Range for minimum samples at a leaf node"
    )
    max_features: list[Literal["sqrt", "log2", None]] = Field(
        ..., description="Options for max features"
    )
    bootstrap: list[bool] = Field(..., description="Options for bootstrap sampling")
    n_trials: int = Field(
        ..., description="Number of Optuna trials for hyperparameter optimization"
    )


class XGBoostConfig(BaseSchema):
    objective: str = Field(..., description="Objective function")
    n_estimators: int = Field(..., description="Number of trees in the ensemble")
    learning_rate: float = Field(..., description="Learning rate")
    early_stopping_rounds: int | None = Field(None, description="Early stopping rounds")
    num_boost_round: int = Field(500, description="Number of boosting rounds")


class XGBoostOptunaConfig(BaseSchema):
    objective: str = Field(..., description="Objective function")
    eval_metric: str = Field(..., description="Evaluation metric")
    booster: list[Literal["gbtree", "gblinear", "dart"]] = Field(
        ..., description="Booster type"
    )
    reg_lambda: tuple[float, float] = Field(
        ..., description="Range for L2 regularization term on weights"
    )
    reg_alpha: tuple[float, float] = Field(
        ..., description="Range for L1 regularization term on weights"
    )
    gamma: tuple[float, float] = Field(
        ..., description="Range for L2 regularization term on weights"
    )
    max_depth: tuple[int, int] = Field(
        ..., description="Range for maximum depth of the trees"
    )
    subsample: tuple[float, float] = Field(
        ..., description="Range for subsample ratio of the training instances"
    )
    min_child_weight: tuple[int, int] = Field(
        ..., description="Range for minimum sum of instance weight needed in a child"
    )
    eta: tuple[float, float] = Field(..., description="Range for learning rate")
    n_estimators: tuple[int, int] = Field(
        ..., description="Range for number of estimators"
    )
    grow_policy: tuple[
        Literal["depthwise", "lossguide"], Literal["depthwise", "lossguide"]
    ] = Field(..., description="Options for grow policy")
    early_stopping_rounds: tuple[int, int] = Field(
        ..., description="Range for number of early stopping rounds"
    )
    n_trials: int = Field(
        ..., description="Number of Optuna trials for hyperparameter optimization"
    )


class LightGBMConfig(BaseSchema):
    objective: str = Field(..., description="Objective function")
    metric: list[str] = Field(..., description="List of evaluation metrics")
    learning_rate: float = Field(..., description="Learning rate")
    early_stopping_rounds: int | None = Field(None, description="Early stopping rounds")
    num_boost_round: int = Field(500, description="Number of boosting rounds")
    verbose: int = Field(1, ge=-1, le=3, description="Verbosity of training")


class LightGBMOptunaConfig(BaseSchema):
    objective: str = Field(..., description="Objective function")
    metric: list[str] = Field(..., description="List of evaluation metrics")
    reg_lambda: tuple[float, float] = Field(
        ..., description="Range for L2 regularization term on weights"
    )
    reg_alpha: tuple[float, float] = Field(
        ..., description="Range for L1 regularization term on weights"
    )
    learning_rate: tuple[float, float] = Field(
        ..., description="Range for learning rate"
    )
    num_leaves: tuple[int, int] = Field(..., description="Range for number of leaves")
    max_depth: tuple[int, int] = Field(
        ..., description="Range for maximum depth of the trees"
    )
    min_child_samples: tuple[int, int] = Field(
        ..., description="Range for minimum child samples"
    )
    n_trials: int = Field(
        ..., description="Number of Optuna trials for hyperparameter optimization"
    )
    num_boost_round: tuple[int, int] = Field(
        ..., description="Range for number of boosting rounds"
    )
    early_stopping_rounds: tuple[int, int] = Field(
        ..., description="Range for number of early stopping rounds"
    )


class ModelTrainingConfig(BaseSchema):
    """Model configuration class."""

    general_config: GeneralConfig = Field(
        ..., description="General model configuration"
    )
    random_forest_config: RandomForestConfig = Field(
        ..., description="Random Forest model configuration"
    )
    xgboost_config: XGBoostConfig = Field(
        ..., description="XGBoost model configuration"
    )
    lightgbm_config: LightGBMConfig = Field(
        ..., description="LightGBM model configuration"
    )


class OptunaConfig(BaseSchema):
    """Optuna hyperparameter optimization configuration class."""

    random_forest_optuna_config: RandomForestOptunaConfig = Field(
        ..., description="Random Forest Optuna configuration"
    )
    xgboost_optuna_config: XGBoostOptunaConfig = Field(
        ..., description="XGBoost Optuna configuration"
    )
    lightgbm_optuna_config: LightGBMOptunaConfig = Field(
        ..., description="LightGBM Optuna configuration"
    )


class Server(BaseSchema):
    """Server configuration class."""

    host: str = Field(..., description="Server host")
    port: int = Field(..., description="Server port")
    workers: int = Field(..., description="Number of worker processes")
    reload: bool = Field(..., description="Enable auto-reload")


class CORS(BaseSchema):
    """CORS configuration class."""

    allow_origins: list[str]
    allow_credentials: bool
    allow_methods: list[str]
    allow_headers: list[str]


class Middleware(BaseSchema):
    """Middleware configuration class."""

    cors: CORS


class APIConfig(BaseSchema):
    """API configuration class."""

    title: str
    name: str
    description: str
    version: str
    status: str
    prefix: str
    server: Server
    middleware: Middleware


class AppConfig(BaseSchema):
    """Application configuration class."""

    feature_config: FeatureConfig = Field(
        description="Feature engineering configuration"
    )
    experiment_config: ExperimentConfig = Field(
        description="Experiment tracking configuration"
    )
    model_training_config: ModelTrainingConfig = Field(
        description="Model training configuration"
    )
    optuna_config: OptunaConfig = Field(
        description="Optuna hyperparameter optimization configuration"
    )
    api_config: APIConfig = Field(description="API configuration")


config_path: Path = PACKAGE_PATH / "src/config/config.yaml"
config: DictConfig = OmegaConf.load(config_path).config
# # Resolve all the variables
resolved_cfg = OmegaConf.to_container(config, resolve=True)
# Validate the config
app_config: AppConfig = AppConfig(**dict(resolved_cfg))  # type: ignore
