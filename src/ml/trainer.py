from datetime import datetime
from typing import Any

import numpy as np
import optuna
import pandas as pd
import polars as pl
import xgboost as xgb
from narwhals.typing import IntoDataFrameT
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit

from src import PACKAGE_PATH, create_logger
from src.config import app_config
from src.exp_tracking.mlflow import ArtifactsType, MLFlowTracker
from src.exp_tracking.mlflow_s3_utils import MLflowS3Manager
from src.exp_tracking.s3_verification import (
    log_s3_verification_results,
    verify_s3_artifacts,
)
from src.ml.utils import (
    compute_metrics,
    create_metrics_df,
    cross_validate_sklearn_model,
    get_feature_importance_from_booster,
    get_model_feature_importance,
    split_into_train_test,
    split_temporal_data,
)
from src.ml.visualization import create_grouped_metrics_barchart
from src.utilities.service_discovery import get_mlflow_endpoint

logger = create_logger(name="trainer")


class ModelTrainer:
    def __init__(self, data: IntoDataFrameT) -> None:
        self.data: IntoDataFrameT = data

        self.mlflow_tracker = MLFlowTracker(
            tracking_uri=get_mlflow_endpoint(),
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

    @property
    def run_name(self) -> str:
        """
        Property to get the run name.

        Returns
        -------
        str
            Run name.
        """
        return self.mlflow_tracker._get_run_name(None)

    def _prepare_data(self) -> dict[str, Any]:
        """Prepare data by applying feature engineering."""
        data_df: IntoDataFrameT = self.data
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

        logger.info(
            "Starting Random Forest training with TimeSeriesSplit cross-validation."
        )
        cv_results: dict[str, Any] = cross_validate_sklearn_model(
            tscv, x_train, y_train, rf_reg
        )

        return cv_results

    def _train_xgboost(self) -> dict[str, Any]:
        """Train an XGBoost model with time series cross-validation."""
        data_dict: dict[str, Any] = self.data_dict

        x_train, y_train = data_dict["x_train"], data_dict["y_train"]
        x_val, y_val = data_dict["x_test"], data_dict["y_test"]
        dtrain = xgb.DMatrix(x_train, y_train, enable_categorical=True)
        dvalid = xgb.DMatrix(x_val, y_val, enable_categorical=True)

        params: dict[str, Any] = (
            app_config.model_training_config.xgboost_config.model_dump()
        )
        params["seed"] = self.random_seed
        early_stopping_rounds: int | None = params.pop("early_stopping_rounds", None)
        num_boost_round: int = params.pop("num_boost_round", 500)

        # Cross-validation
        cv_results: pd.DataFrame = xgb.cv(
            params=params,
            dtrain=dtrain,
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
            dtrain=dtrain,
            num_boost_round=best_num_rounds,
        )
        preds = final_model.predict(dvalid)
        metrics: dict[str, float | None] = compute_metrics(
            y_val, preds, n_features=x_train.shape[1]
        )

        return {
            "model": final_model,
            "metrics": {
                "RMSE": metrics.get("RMSE"),
                "MAE": metrics.get("MAE"),
                "MAPE": metrics.get("MAPE"),
                "Adjusted_R2": metrics.get("Adjusted_R2"),
            },
        }

    def _train_lightgbm(self) -> dict[str, Any]:
        """Train a LightGBM model with time series cross-validation."""
        return {}

    def _hyperparameter_tuning_lightgbm(self) -> None:
        pass

    def train_all_models(self, tune_models: bool) -> list[dict[str, Any]]:
        """Train models and log results to MLflow."""
        column_names = self.data_dict["columns"]
        results: list[dict[str, Any]] = []
        data_dict: dict[str, Any] = self.data_dict

        if tune_models:
            logger.info("🔧 Training with hyperparameter tuning enabled")
            return self._hyperparameter_tuning_all_models()

        logger.info("🚀 Training with default hyperparameters")
        try:
            with self.mlflow_tracker.start_run() as run_id:
                # Log data stats
                self.mlflow_tracker.log_params(
                    {
                        "train_size": len(data_dict["x_train"]),
                        "test_size": len(data_dict["x_test"]),
                        "n_features": data_dict["x_train"].shape[1],
                    }
                )

                # ================================
                # ====== Train Random Forest =====
                # ================================
                rf_reg_results = self._train_random_forest()
                y_pred: np.ndarray = rf_reg_results.get("model").predict(  # type: ignore
                    data_dict["x_test"]
                )

                # Feature importance
                rf_model: RandomForestRegressor = rf_reg_results.get("model")
                weights_ = rf_model.feature_importances_
                model_name: str = type(rf_model).__name__

                tags: dict[str, Any]
                (_, tags) = get_model_feature_importance(
                    model_name=model_name, features=column_names, weights=weights_, n=20
                )
                self.mlflow_tracker.log_mlflow_artifact(
                    object=tags,
                    object_type=ArtifactsType.JSON,
                    filename="feat_imp_random_forest",
                    artifact_dest="models/",
                )
                self.mlflow_tracker.log_model(
                    model=rf_model,
                    model_name="random_forest_model",
                    input_example=self.input_example,
                    save_format=ArtifactsType.PICKLE,
                )
                self.mlflow_tracker.log_metrics(rf_reg_results.get("metrics"))  # type: ignore
                results.append(
                    {
                        "run_id": run_id,
                        "model_name": model_name,
                        "model": rf_model,
                        "metrics": rf_reg_results.get("metrics"),
                        "predictions": y_pred,
                    }
                )
                logger.info("Random Forest training completed successfully.")

                # ================================
                # ========= Train XGBoost ========
                # ================================
                xgb_results: dict[str, Any] = self._train_xgboost()
                y_pred = xgb_results.get("model").predict(  # type: ignore
                    xgb.DMatrix(data_dict["x_test"], enable_categorical=True)
                )

                # Feature importance
                xgb_model: xgb.Booster = xgb_results.get("model")
                model_name = type(xgb_model).__name__
                weights_dict: dict[str, float] = get_feature_importance_from_booster(
                    xgb_model,
                    column_names,
                    importance_type="weight",
                )
                weights_ = list(weights_dict.values())

                _, tags = get_model_feature_importance(
                    model_name=model_name, features=column_names, weights=weights_, n=20
                )
                self.mlflow_tracker.log_mlflow_artifact(
                    object=tags,
                    object_type=ArtifactsType.JSON,
                    filename="feat_imp_xgboost",
                    artifact_dest="models/",
                )
                self.mlflow_tracker.log_model(
                    model=xgb_model,
                    model_name="xgboost_model",
                    input_example=self.input_example,
                    save_format=ArtifactsType.JSON,
                )
                self.mlflow_tracker.log_metrics(xgb_results.get("metrics"))  # type: ignore

                results.append(
                    {
                        "run_id": run_id,
                        "model_name": model_name,
                        "model": xgb_model,
                        "metrics": xgb_results.get("metrics"),
                        "predictions": y_pred,
                    }
                )
                logger.info("XGBoost training completed successfully.")

                logger.info("✅ Model training completed successfully.")
                test_df: pl.DataFrame = pl.DataFrame(
                    data_dict["x_test"], schema=column_names
                )
                self.generate_and_log_visualizations(results=results, test_df=test_df)
                self.mlflow_tracker.log_artifact_from_path(
                    local_path=PACKAGE_PATH / "reports",
                    artifact="visualizations",
                    delete_tmp=True,
                )

            self.mlflow_tracker.end_run()

            if run_id:
                try:
                    logger.info("Syncing artifacts to S3...")
                    s3_manager = MLflowS3Manager()
                    s3_manager.sync_mlflow_artifacts_to_s3(run_id)
                    logger.info("✅ Successfully synced artifacts to S3")

                    # Verify S3 artifacts after sync
                    logger.info("Verifying S3 artifact storage...")
                    verification_results: dict[str, Any] = verify_s3_artifacts(
                        run_id=run_id,
                        expected_artifacts=[
                            "visualizations/",
                            "models/",
                        ],
                    )
                    log_s3_verification_results(verification_results)

                    if not verification_results["success"]:
                        logger.warning("S3 artifact verification failed after sync")
                except Exception as e:
                    logger.error(f"❌ Failed to sync artifacts to S3: {e}")

            return results

        except Exception as e:
            logger.error(f"❌ Error during model training: {e}")
            raise

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

    def _objective_random_forest(self, trial: optuna.Trial) -> float:
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
            metrics: dict[str, float | None] = cv_results.get("metrics", {})

            # Return mean RMSE across all CV folds
            mean_rmse = metrics.get("RMSE", 0.0)
            print(f"Trial {trial.number}: Mean RMSE = {mean_rmse}")

            # Store additional metrics for analysis
            trial.set_user_attr("RMSE", mean_rmse)
            trial.set_user_attr("MAE", metrics.get("MAE"))
            trial.set_user_attr("MAPE", metrics.get("MAPE"))
            trial.set_user_attr("Adjusted_R2", metrics.get("Adjusted_R2"))

        return mean_rmse  # type: ignore

    def _objective_xgboost(self, trial: optuna.Trial) -> float:
        with self.mlflow_tracker.start_run(nested=True):
            optuna_config = app_config.optuna_config.xgboost_optuna_config
            # Define hyperparameters
            params = {
                "objective": "reg:squarederror",
                "eval_metric": optuna_config.eval_metric,
                "booster": trial.suggest_categorical("booster", optuna_config.booster),
                "lambda": trial.suggest_float(
                    "lambda_",
                    optuna_config.lambda_[0],
                    optuna_config.lambda_[1],
                    log=True,
                ),
                "alpha": trial.suggest_float(
                    "alpha", optuna_config.alpha[0], optuna_config.alpha[1], log=True
                ),
            }

            if params["booster"] == "gbtree" or params["booster"] == "dart":
                params["max_depth"] = trial.suggest_int(
                    "max_depth", optuna_config.max_depth[0], optuna_config.max_depth[1]
                )
                params["eta"] = trial.suggest_float(
                    "eta", optuna_config.eta[0], optuna_config.eta[1], log=True
                )
                params["gamma"] = trial.suggest_float(
                    "gamma", optuna_config.gamma[0], optuna_config.gamma[1], log=True
                )
                params["grow_policy"] = trial.suggest_categorical(
                    "grow_policy", optuna_config.grow_policy
                )

            data_dict: dict[str, Any] = self.data_dict
            x_train, y_train = data_dict["x_train"], data_dict["y_train"]
            x_val, y_val = data_dict["x_test"], data_dict["y_test"]
            dtrain = xgb.DMatrix(x_train, y_train, enable_categorical=True)
            dvalid = xgb.DMatrix(x_val, y_val, enable_categorical=True)

            # Train XGBoost model
            bst = xgb.train(params, dtrain)
            preds = bst.predict(dvalid)
            metrics: dict[str, float | None] = compute_metrics(
                y_val, preds, n_features=x_train.shape[1]
            )

            mean_rmse = metrics.get("RMSE", 0.0)
            mean_mae = metrics.get("MAE", 0.0)
            mean_mape = metrics.get("MAPE", 0.0)
            mean_adj_r2 = metrics.get("Adjusted_R2", 0.0)
            print(f"Trial {trial.number}: Mean RMSE = {mean_rmse}")

            # Store additional metrics for analysis
            trial.set_user_attr("RMSE", mean_rmse)
            trial.set_user_attr("MAE", mean_mae)
            trial.set_user_attr("MAPE", mean_mape)
            trial.set_user_attr("Adjusted_R2", mean_adj_r2)

        return mean_rmse  # type: ignore

    def _hyperparameter_tuning_random_forest(self) -> dict[str, Any]:
        """
        Perform hyperparameter tuning for a RandomForestRegressor using Optuna and log results to MLflow.
        """
        with self.mlflow_tracker.start_run(nested=True) as run_id:
            study = optuna.create_study(
                direction="minimize",
                sampler=optuna.samplers.TPESampler(seed=self.random_seed),
            )
            study.optimize(
                self._objective_random_forest,
                n_trials=self.n_trials,
                callbacks=[self.champion_callback],
            )
            print("Best trial:")
            best_trial = study.best_trial
            best_params: dict[str, Any] = best_trial.params
            rmse: float = best_trial.user_attrs.get("RMSE", 0.0)
            mae: float = best_trial.user_attrs.get("MAE", 0.0)
            mape: float = best_trial.user_attrs.get("MAPE", 0.0)
            adj_r2: float | None = best_trial.user_attrs.get("Adjusted_R2", None)

            # Log best parameters and metrics to MLflow
            self.mlflow_tracker.log_params(best_params)
            self.mlflow_tracker.log_metrics({"best_rmse": best_trial.value})
            self.mlflow_tracker.log_metrics({"mean_rmse": rmse})
            self.mlflow_tracker.log_metrics({"mean_mae": mae})
            self.mlflow_tracker.log_metrics({"mean_mape": mape})
            if adj_r2 is not None:
                self.mlflow_tracker.log_metrics({"mean_adjusted_r2": adj_r2})

            # Log tags
            tags: dict[str, Any] = (
                app_config.experiment_config.experiment_tags.model_dump()
            )
            tags["model_family"] = "RandomForest"
            self.mlflow_tracker.set_tags(tags=tags)

            rf_reg = RandomForestRegressor(
                **best_params,
            )
            self.mlflow_tracker.log_model(
                rf_reg,
                model_name="RandomForestRegressor",
                input_example=self.input_example,
            )

        return {
            "run_id": run_id,
            "model_name": "RandomForestRegressor",
            "best_params": best_params,
        }

    def _hyperparameter_tuning_xgboost(self) -> dict[str, Any]:
        with self.mlflow_tracker.start_run(nested=True) as run_id:
            study = optuna.create_study(
                direction="minimize",
                sampler=optuna.samplers.TPESampler(seed=self.random_seed),
            )
            study.optimize(
                self._objective_xgboost,
                n_trials=self.n_trials,
                callbacks=[self.champion_callback],
            )
            print("Best trial:")
            best_trial = study.best_trial
            best_params: dict[str, Any] = best_trial.params
            rmse: list[float] | float = best_trial.user_attrs.get("RMSE", [])
            mae: list[float] | float = best_trial.user_attrs.get("MAE", [])
            mape: list[float] | float = best_trial.user_attrs.get("MAPE", [])
            adj_r2: list[float | None] = best_trial.user_attrs.get("Adjusted_R2", [])

            # Log best parameters and metrics to MLflow
            self.mlflow_tracker.log_params(best_params)
            self.mlflow_tracker.log_metrics({"best_rmse": best_trial.value})
            self.mlflow_tracker.log_metrics({"mean_rmse": rmse})  # type: ignore
            self.mlflow_tracker.log_metrics({"mean_mae": mae})  # type: ignore
            self.mlflow_tracker.log_metrics({"mean_mape": mape})  # type: ignore
            self.mlflow_tracker.log_metrics({"mean_adjusted_r2": adj_r2})  # type: ignore

            # Log tags
            tags: dict[str, Any] = (
                app_config.experiment_config.experiment_tags.model_dump()
            )
            tags["model_family"] = "XGBoost"
            self.mlflow_tracker.set_tags(tags=tags)

            # Build model
            data_dict: dict[str, Any] = self.data_dict
            x_train, y_train = data_dict["x_train"], data_dict["y_train"]
            dtrain = xgb.DMatrix(x_train, y_train, enable_categorical=True)
            model = xgb.train(best_params, dtrain)
            self.mlflow_tracker.log_model(
                model,
                model_name="XGBoostRegressor",
                input_example=self.input_example,
                save_format=ArtifactsType.JSON,
            )
        return {
            "run_id": run_id,
            "model_name": "XGBoostRegressor",
            "best_params": best_params,
        }

    def _hyperparameter_tuning_all_models(self) -> list[dict[str, Any]]:
        """Perform hyperparameter tuning for all models and log results to MLflow."""
        results: list[dict[str, Any]] = []

        try:
            logger.info("🚨 Starting hyperparameter tuning for all models...")
            # ================================
            # ========= Random Forest ========
            # ================================
            rf_results: dict[str, Any] = self._hyperparameter_tuning_random_forest()

            # ================================
            # ============ XGBoost ===========
            # ================================
            xgb_results: dict[str, Any] = self._hyperparameter_tuning_xgboost()
            results.extend([rf_results, xgb_results])
            logger.info("✅ Hyperparameter tuning completed successfully.")

            return results

        except Exception as e:
            logger.error(f"❌ Error during hyperparameter tuning: {e}")
            raise

    def generate_and_log_visualizations(
        self, results: list[dict[str, Any]], test_df: pl.DataFrame
    ) -> None:
        """Generate and log model comparison visualizations to MLflow"""
        date_str: str = datetime.now().isoformat(timespec="seconds")
        try:
            results_df = create_metrics_df(results)
            create_grouped_metrics_barchart(
                results_df, save_path=f"model_metrics_comparison_{date_str}.html"
            )
            logger.info("✅ Successfully generated visualizations.")

        except Exception as e:
            # Don't fail the entire run
            logger.error(f"❌ Failed to generate visualizations: {e}")
