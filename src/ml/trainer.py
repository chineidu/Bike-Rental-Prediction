from typing import Any

import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from narwhals.typing import IntoDataFrameT
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit

from src import create_logger
from src.config import app_config
from src.config.config import FeatureConfig
from src.exp_tracking.mlflow import MLFlowTracker
from src.ml.feature_engineering import FeatureEngineer
from src.ml.utils import (
    cross_validate_sklearn_model,
    split_into_train_test,
    split_temporal_data,
)

logger = create_logger(name="trainer")


class ModelTrainer:
    def __init__(self, data: IntoDataFrameT, config: FeatureConfig) -> None:
        self.data: IntoDataFrameT = data
        self.config: FeatureConfig = config

        self.feat_eng = FeatureEngineer()
        self.mlflow_tracker = MLFlowTracker(
            tracking_uri=app_config.experiment_config.tracking_uri,
            experiment_name=app_config.experiment_config.experiment_name,
        )
        self.input_example: IntoDataFrameT | None = None
        self.data_dict: dict[str, Any] = self._prepare_data()

        # Configs
        self.n_splits: int = app_config.model_training_config.general_config.n_splits
        self.cv_test_size: int = (
            app_config.model_training_config.general_config.cv_test_size
        )
        self.random_seed: int = (
            app_config.model_training_config.general_config.random_seed
        )
        self.n_trials: int = app_config.model_training_config.general_config.n_trials

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(data_shape={self.data.shape})"

    def _prepare_data(self) -> dict[str, Any]:
        """Prepare data by applying feature engineering."""
        data_df: IntoDataFrameT = self.feat_eng.create_all_features(
            data=self.data, config=self.config
        )
        train_df, test_df = split_temporal_data(data_df)
        data_dict: dict[str, Any] = split_into_train_test(
            train_df, test_df, target_col="target"
        )

        logger.info("Data preparation complete.")
        self.input_example = train_df
        return data_dict

    def _train_random_forest(self) -> dict[str, Any]:
        """Train a Random Forest model with time series cross-validation."""

        tscv = TimeSeriesSplit(
            n_splits=self.n_splits,
            test_size=self.cv_test_size,
            gap=0,
        )
        rf_reg = RandomForestRegressor(
            **app_config.model_training_config.random_forest_config.model_dump(),
            random_state=self.random_seed,
        )
        data_dict: dict[str, Any] = self.data_dict
        x_train, y_train = data_dict["x_train"], data_dict["y_train"]
        cv_results: dict[str, Any] = cross_validate_sklearn_model(
            tscv, x_train, y_train, rf_reg
        )

        return {
            "model": cv_results.get("model"),
            "metrics": {
                "RMSE": cv_results.get("RMSE"),
                "MAE": cv_results.get("MAE"),
                "MAPE": cv_results.get("MAPE"),
            },
        }

    def _train_xgboost(self) -> dict[str, Any]:
        """Train an XGBoost model with time series cross-validation."""
        data_dict: dict[str, Any] = self.data_dict
        x_train, y_train = data_dict["x_train"], data_dict["y_train"]
        dtrain_reg = xgb.DMatrix(x_train, y_train, enable_categorical=True)
        params: dict[str, Any] = (
            app_config.model_training_config.xgboost_config.model_dump()
        )
        print(params)
        params["seed"] = self.random_seed
        early_stopping_rounds: int | None = params.pop("early_stopping_rounds", None)
        num_boost_round: int = params.pop("num_boost_round", 500)

        # Cross-validation
        cv_results: pd.DataFrame = xgb.cv(
            params=params,
            dtrain=dtrain_reg,
            num_boost_round=num_boost_round,
            nfold=self.n_splits,
            metrics={"rmse"},
            seed=self.random_seed,
            as_pandas=True,
            callbacks=[
                xgb.callback.EvaluationMonitor(show_stdv=True),  # type: ignore
                xgb.callback.EarlyStopping(rounds=early_stopping_rounds),  # type: ignore
            ],
        )
        best_num_rounds: int = len(cv_results)
        final_model: xgb.Booster = xgb.train(
            params=params,
            dtrain=dtrain_reg,
            num_boost_round=best_num_rounds,
        )

        return {
            "model": final_model,
            "metrics": {
                "RMSE": cv_results["test-rmse-mean"].tolist(),
                "MAE": None,  # Placeholder, implement if needed
                "MAPE": None,  # Placeholder, implement if needed
            },
        }

    def _train_lightgbm(self) -> dict[str, Any]:
        """Train a LightGBM model with time series cross-validation."""
        return {}

    def _hyperparameter_tuning_xgboost(self) -> None:
        pass

    def _hyperparameter_tuning_lightgbm(self) -> None:
        pass

    def train_model(self) -> None:
        return

    @staticmethod
    def champion_callback(
        study: optuna.study.Study, frozen_trial: optuna.trial.FrozenTrial
    ) -> None:
        """
        Callback that logs when a trial improves the current best value for an Optuna study.

        Parameters
        ----------
        study : optuna.study.Study
            The Optuna Study object being optimized.
        frozen_trial : optuna.trial.FrozenTrial
            The completed trial.

        Returns
        -------
        None
            This callback does not return a value.

        Notes
        -----
        This function updates ``study.user_attrs['winner']`` when a new best value is found and
        logs either an initial-trial message or the percentage improvement; avoid using this
        callback in distributed execution environments due to potential race conditions when
        reading/updating ``study.user_attrs``.
        """
        winner = study.user_attrs.get("winner", None)

        # Only respond when there is a best value to compare
        if study.best_value is not None and winner != study.best_value:
            study.set_user_attr("winner", study.best_value)
            if winner is not None:
                # protect against division by zero
                if study.best_value == 0:
                    improvement_percent = float("inf")
                else:
                    improvement_percent = (
                        abs(winner - study.best_value) / abs(study.best_value)
                    ) * 100
                logger.info(
                    f"Trial {frozen_trial.number} achieved value: {frozen_trial.value} with "
                    f"{improvement_percent:.4f}% improvement",
                )
            else:
                logger.info(
                    f"Initial trial {frozen_trial.number} achieved value: {frozen_trial.value}"
                )

    def objective_random_forest(self, trial: optuna.Trial) -> float:
        """Objective function for Optuna hyperparameter optimization using TimeSeriesSplit."""
        with self.mlflow_tracker.start_run(nested=True):
            tscv = TimeSeriesSplit(
                n_splits=self.n_splits,
                test_size=self.cv_test_size,
                gap=0,
            )
            optuna_config = app_config.optuna_config.random_forest_optuna_config
            params: dict[str, Any] = {
                "n_estimators": trial.suggest_int(
                    "n_estimators",
                    optuna_config.n_estimators[0],
                    optuna_config.n_estimators[1],
                ),
                "max_depth": trial.suggest_int(
                    "max_depth", optuna_config.max_depth[0], optuna_config.max_depth[1]
                ),
                "min_samples_split": trial.suggest_int(
                    "min_samples_split",
                    optuna_config.min_samples_split[0],
                    optuna_config.min_samples_split[1],
                ),
                "min_samples_leaf": trial.suggest_int(
                    "min_samples_leaf",
                    optuna_config.min_samples_leaf[0],
                    optuna_config.min_samples_leaf[1],
                ),
                "max_features": trial.suggest_categorical(
                    "max_features", optuna_config.max_features
                ),
                "bootstrap": trial.suggest_categorical(
                    "bootstrap", optuna_config.bootstrap
                ),
                "random_state": self.random_seed,
            }
            rf_reg = RandomForestRegressor(**params)
            data_dict: dict[str, Any] = self.data_dict
            x_train, y_train = data_dict["x_train"], data_dict["y_train"]

            # Use TimeSeriesSplit for cross-validation
            cv_results: dict[str, Any] = cross_validate_sklearn_model(
                tscv, x_train, y_train, rf_reg
            )

            # Return mean RMSE across all CV folds
            mean_rmse = np.mean(cv_results.get("RMSE"))
            print(f"Trial {trial.number}: Mean RMSE = {mean_rmse}")

            # Store additional metrics for analysis
            trial.set_user_attr("RMSE", cv_results.get("RMSE"))
            trial.set_user_attr("RMSE_std", np.std(cv_results.get("RMSE")).round(4))
            trial.set_user_attr("MAE", cv_results.get("MAE"))
            trial.set_user_attr("MAPE", cv_results.get("MAPE"))

        return mean_rmse  # type: ignore

    def _hyperparameter_tuning_random_forest(self) -> None:
        """
        Perform hyperparameter tuning for a RandomForestRegressor using Optuna and log results to MLflow.
        """
        with self.mlflow_tracker.start_run(nested=True):
            study = optuna.create_study(
                direction="minimize",
                sampler=optuna.samplers.TPESampler(seed=self.random_seed),
            )
            study.optimize(
                self.objective_random_forest,
                n_trials=self.n_trials,
                callbacks=[self.champion_callback],
            )
            print("Best trial:")
            best_trial = study.best_trial
            best_params: dict[str, Any] = best_trial.params
            rmse: list[float] = best_trial.user_attrs.get("RMSE", [])
            rmse_std: float = best_trial.user_attrs.get("RMSE_std", 0.0)
            mae: list[float] = best_trial.user_attrs.get("MAE", [])
            mape: list[float] = best_trial.user_attrs.get("MAPE", [])

            # Log best parameters and metrics to MLflow
            self.mlflow_tracker.log_params(best_params)
            self.mlflow_tracker.log_metrics({"best_rmse": best_trial.value})
            self.mlflow_tracker.log_metrics({"std_rmse": rmse_std})
            self.mlflow_tracker.log_metrics({"mean_rmse": np.mean(rmse).round(4)})
            self.mlflow_tracker.log_metrics({"mean_mae": np.mean(mae).round(4)})
            self.mlflow_tracker.log_metrics({"mean_mape": np.mean(mape).round(4)})

            # Log tags
            self.mlflow_tracker.set_tags(
                tags={
                    "project": self.mlflow_tracker.experiment_name,
                    "team": "mlops",
                    "optimizer_engine": "optuna",
                    "model_family": "RandomForest",
                    "feature_set_version": 1,
                }
            )

            rf_reg = RandomForestRegressor(
                **best_params,
            )
            self.mlflow_tracker.log_model(
                rf_reg,
                model_name="RandomForestRegressor",
                input_example=self.input_example,
            )
