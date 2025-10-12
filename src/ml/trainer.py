from datetime import datetime
from typing import Any

import lightgbm as lgb
import numpy as np
import optuna
import polars as pl
import xgboost as xgb
from narwhals.typing import IntoDataFrameT
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit

from src import PACKAGE_PATH, create_logger
from src.config import app_config
from src.exceptions import HyperparameterTuningError, TrainingError
from src.exp_tracking.mlflow import ArtifactsType, MLFlowTracker
from src.exp_tracking.mlflow_s3_utils import MLflowS3Manager
from src.exp_tracking.s3_verification import (
    log_s3_verification_results,
    verify_s3_artifacts,
)
from src.ml.utils import (
    combine_train_val,
    compute_metrics,
    create_metrics_df,
    cross_validate_sklearn_model,
    get_feature_importance_from_booster,
    get_lightgbm_feature_importance,
    get_model_feature_importance,
    split_into_train_val_test_sets,
)
from src.ml.visualization import create_grouped_metrics_barchart
from src.schemas.types import ModelType
from src.utilities.service_discovery import get_mlflow_endpoint

logger = create_logger(name="trainer")


class ModelTrainer:
    def __init__(self, data: IntoDataFrameT) -> None:
        self.data: IntoDataFrameT = data
        self.input_example: IntoDataFrameT = data

        self.mlflow_tracker = MLFlowTracker(
            tracking_uri=get_mlflow_endpoint(),
            experiment_name=app_config.experiment_config.experiment_name,
        )
        self.test_size: float | int = (
            app_config.model_training_config.general_config.test_size
        )
        self.data_dict: dict[str, Any] = self._prepare_data()

        # Configs
        self.n_splits: int = app_config.model_training_config.general_config.n_splits
        self.cv_test_size: int = (
            app_config.model_training_config.general_config.cv_test_size
        )
        self.random_seed: int = (
            app_config.model_training_config.general_config.random_seed
        )

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
        data_dict: dict[str, Any] = split_into_train_val_test_sets(
            data_df, test_size=self.test_size, target_col="target"
        )

        logger.info("Data preparation complete.")
        return data_dict

    def _train_random_forest(self, params: dict[str, Any]) -> dict[str, Any]:
        """Train a Random Forest model with time series cross-validation."""

        tscv = TimeSeriesSplit(
            n_splits=self.n_splits,
            test_size=self.cv_test_size,
            gap=0,
        )
        params["random_state"] = self.random_seed
        model = RandomForestRegressor(**params)
        data_dict: dict[str, Any] = self.data_dict
        x_train, y_train = data_dict["x_train"], data_dict["y_train"]
        x_val, y_val = data_dict["x_val"], data_dict["y_val"]
        X, y = combine_train_val(x_train, y_train, x_val, y_val)

        logger.info(
            "Starting Random Forest training with TimeSeriesSplit cross-validation."
        )
        cv_results: dict[str, Any] = cross_validate_sklearn_model(tscv, X, y, model)

        return cv_results

    def _train_xgboost(self, params: dict[str, Any]) -> dict[str, Any]:
        """Train an XGBoost model with time series cross-validation."""
        data_dict: dict[str, Any] = self.data_dict

        x_train, y_train = data_dict["x_train"], data_dict["y_train"]
        x_val, y_val = data_dict["x_val"], data_dict["y_val"]

        dtrain = xgb.DMatrix(x_train, y_train, enable_categorical=True)
        dval = xgb.DMatrix(x_val, y_val, enable_categorical=True)

        params["seed"] = self.random_seed
        early_stopping_rounds: int = params.pop("early_stopping_rounds", 10)
        num_boost_round: int = params.pop("num_boost_round", 500)

        # Train the model
        model: xgb.Booster = xgb.train(
            params=params,
            dtrain=dtrain,
            num_boost_round=num_boost_round,
            verbose_eval=10,
            evals=[(dval, "validation")],
            early_stopping_rounds=early_stopping_rounds,
        )
        preds = model.predict(dval)
        metrics: dict[str, float | None] = compute_metrics(
            y_val, preds, n_features=x_train.shape[1]
        )

        return {
            "model": model,
            "metrics": {
                "RMSE": metrics.get("RMSE"),
                "MAE": metrics.get("MAE"),
                "MAPE": metrics.get("MAPE"),
                "Adjusted_R2": metrics.get("Adjusted_R2"),
            },
        }

    def _train_lightgbm(self, params: dict[str, Any]) -> dict[str, Any]:
        """Train a LightGBM model."""
        data_dict: dict[str, Any] = self.data_dict

        x_train, y_train = data_dict["x_train"], data_dict["y_train"]
        x_val, y_val = data_dict["x_val"], data_dict["y_val"]

        # Create datasets for LightGBM
        lgb_train = lgb.Dataset(x_train, y_train)
        lgb_val = lgb.Dataset(x_val, y_val, reference=lgb_train)

        params["seed"] = self.random_seed
        num_boost_round: int = params.pop("num_boost_round", 500)
        early_stopping_rounds: int = params.pop("early_stopping_rounds", 50)

        model = lgb.train(
            params,
            lgb_train,
            num_boost_round=num_boost_round,
            valid_sets=[lgb_val],
            valid_names=["validation"],
            callbacks=[
                lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=True),
            ],
        )
        preds = model.predict(x_val)
        metrics: dict[str, float | None] = compute_metrics(
            y_val,
            preds,  # type: ignore
            n_features=x_train.shape[1],
        )

        return {
            "model": model,
            "metrics": {
                "RMSE": metrics.get("RMSE"),
                "MAE": metrics.get("MAE"),
                "MAPE": metrics.get("MAPE"),
                "Adjusted_R2": metrics.get("Adjusted_R2"),
            },
        }

    def train_all_models(self) -> list[dict[str, Any]]:
        """Train models and log results to MLflow."""
        column_names = self.data_dict["columns"]
        results: list[dict[str, Any]] = []
        data_dict: dict[str, Any] = self.data_dict

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
                logger.info("Training Random Forest ...")
                params_rf: dict[str, Any] = (
                    app_config.model_training_config.random_forest_config.model_dump()
                )
                rf_reg_results = self._train_random_forest(params=params_rf)
                y_pred: np.ndarray = rf_reg_results["model"].predict(
                    data_dict["x_test"]
                )

                # Feature importance
                rf_model: RandomForestRegressor = rf_reg_results.get("model")
                weights_ = rf_model.feature_importances_
                # model_name: str = str(ModelType.RANDOM_FOREST)
                model_name = ModelType.RANDOM_FOREST

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
                    model_name=model_name,
                    input_example=self.input_example,
                    save_format=ArtifactsType.PICKLE,
                )
                metrics: dict[str, float | None] = self._format_metrics(  # type: ignore
                    rf_reg_results.get("metrics", {}), prefix="rf"
                )
                self.mlflow_tracker.log_metrics(metrics)  # type: ignore
                results.append(
                    {
                        "run_id": run_id,
                        "model_name": model_name,
                        "model": rf_model,
                        "metrics": rf_reg_results.get("metrics"),
                        "predictions": y_pred,
                    }
                )
                logger.info("🚀 Random Forest training completed successfully.")

                # ================================
                # ========= Train XGBoost ========
                # ================================
                logger.info("Training XGBoost ...")
                params_xgb: dict[str, Any] = (
                    app_config.model_training_config.xgboost_config.model_dump()
                )
                xgb_results: dict[str, Any] = self._train_xgboost(params=params_xgb)
                y_pred = xgb_results["model"].predict(
                    xgb.DMatrix(data_dict["x_test"], enable_categorical=True)
                )

                # Feature importance
                xgb_model: xgb.Booster = xgb_results.get("model")
                model_name = ModelType.XGBOOST

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
                    model_name=model_name,
                    input_example=self.input_example,
                    save_format=ArtifactsType.JSON,
                )
                metrics: dict[str, float | None] = self._format_metrics(  # type: ignore
                    xgb_results.get("metrics", {}), prefix="xgb"
                )
                self.mlflow_tracker.log_metrics(metrics)  # type: ignore

                results.append(
                    {
                        "run_id": run_id,
                        "model_name": model_name,
                        "model": xgb_model,
                        "metrics": xgb_results.get("metrics"),
                        "predictions": y_pred,
                    }
                )
                logger.info("🚀 XGBoost training completed successfully.")

                # ================================
                # ========= Train LightGBM =======
                # ================================
                logger.info("Training LightGBM ...")
                params_lgb: dict[str, Any] = (
                    app_config.model_training_config.lightgbm_config.model_dump()
                )
                lgb_results: dict[str, Any] = self._train_lightgbm(params=params_lgb)
                y_pred = lgb_results["model"].predict(data_dict["x_test"])

                # Feature importance
                lgb_model: lgb.Booster = lgb_results.get("model")
                model_name = ModelType.LIGHTGBM
                feat_imp_df: pl.DataFrame = get_lightgbm_feature_importance(
                    lgb_model, column_names
                )
                weights_ = feat_imp_df["gain_importance_normalized"].to_list()

                _, tags = get_model_feature_importance(
                    model_name=model_name, features=column_names, weights=weights_, n=20
                )
                self.mlflow_tracker.log_mlflow_artifact(
                    object=tags,
                    object_type=ArtifactsType.JSON,
                    filename="feat_imp_lightgbm",
                    artifact_dest="models/",
                )
                self.mlflow_tracker.log_model(
                    model=lgb_model,
                    model_name=model_name,
                    input_example=self.input_example,
                    save_format=ArtifactsType.TXT,
                )
                metrics = self._format_metrics(  # type: ignore
                    lgb_results.get("metrics", {}), prefix="lgb"
                )
                self.mlflow_tracker.log_metrics(metrics)  # type: ignore

                results.append(
                    {
                        "run_id": run_id,
                        "model_name": model_name,
                        "model": lgb_model,
                        "metrics": lgb_results.get("metrics"),
                        "predictions": y_pred,
                    }
                )
                logger.info("🚀 LightGBM training completed successfully.")

                logger.info("✅ ALL models training completed successfully.")
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

        except TrainingError as e:
            logger.error(f"❌ Error during model training: {e}")
            raise

    @staticmethod
    def champion_callback(
        study: optuna.study.Study, frozen_trial: optuna.trial.FrozenTrial
    ) -> None:
        """
        Callback that logs ONLY when a new best value is achieved.

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
        This function only logs when a trial improves upon the current best value.
        It updates study.user_attrs['previous_best'] to track improvements.
        """
        current_best: float | None = study.best_value
        trial_value: float | None = frozen_trial.value
        trial_number: int = frozen_trial.number

        # Only log if this trial achieved a new best value
        if trial_value is not None and current_best == trial_value:
            # Get the previous best value from user attributes
            previous_best: float | None = study.user_attrs.get("previous_best", None)

            if previous_best is None:
                # First trial (initial champion)
                logger.info(
                    f"Initial trial {trial_number} achieved value: {trial_value}"
                )
            else:
                # Calculate improvement percentage
                if abs(previous_best) < 1e-10:  # Handle near-zero values
                    improvement_percent = (
                        float("inf") if previous_best != current_best else 0.0
                    )
                else:
                    improvement_percent = (
                        (previous_best - current_best) / abs(previous_best)
                    ) * 100

                logger.info(
                    f"Trial {trial_number} achieved value: {trial_value} with {improvement_percent:.4f}% improvement"
                )

            # Update the previous best for next comparison
            study.set_user_attr("previous_best", current_best)

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
            x_val, y_val = data_dict["x_val"], data_dict["y_val"]
            X, y = combine_train_val(x_train, y_train, x_val, y_val)

            # Use TimeSeriesSplit for cross-validation
            cv_results: dict[str, Any] = cross_validate_sklearn_model(
                tscv, X, y, rf_reg
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

            # Log to MLflow
            self.mlflow_tracker.log_params(params)
            self.mlflow_tracker.log_metrics(metrics)  # type: ignore

        return mean_rmse  # type: ignore

    def _objective_xgboost(self, trial: optuna.Trial) -> float:
        with self.mlflow_tracker.start_run(nested=True):
            optuna_config = app_config.optuna_config.xgboost_optuna_config
            # Define hyperparameters
            # NB: Use `log=True` for parameters that span several orders of magnitude
            # e.g , learning_rate, reg_alpha, reg_lambda, etc.
            params = {
                "objective": optuna_config.objective,
                "eval_metric": optuna_config.eval_metric,
                "booster": trial.suggest_categorical("booster", optuna_config.booster),
                "reg_lambda": trial.suggest_float(
                    "reg_lambda",
                    optuna_config.reg_lambda[0],
                    optuna_config.reg_lambda[1],
                    log=True,
                ),
                "reg_alpha": trial.suggest_float(
                    "reg_alpha",
                    optuna_config.reg_alpha[0],
                    optuna_config.reg_alpha[1],
                    log=True,
                ),
                "gamma": trial.suggest_float(
                    "gamma", optuna_config.gamma[0], optuna_config.gamma[1], log=True
                ),
                "min_child_weight": trial.suggest_int(
                    "min_child_weight",
                    optuna_config.min_child_weight[0],
                    optuna_config.min_child_weight[1],
                ),
                "subsample": trial.suggest_float(
                    "subsample", optuna_config.subsample[0], optuna_config.subsample[1]
                ),
                "eta": trial.suggest_float(
                    "eta", optuna_config.eta[0], optuna_config.eta[1], log=True
                ),
                "n_estimators": trial.suggest_int(
                    "n_estimators",
                    optuna_config.n_estimators[0],
                    optuna_config.n_estimators[1],
                ),
                "grow_policy": trial.suggest_categorical(
                    "grow_policy", optuna_config.grow_policy
                ),
                "early_stopping_rounds": trial.suggest_int(
                    "early_stopping_rounds",
                    optuna_config.early_stopping_rounds[0],
                    optuna_config.early_stopping_rounds[1],
                ),
                "seed": self.random_seed,
            }

            if params["booster"] == "gbtree" or params["booster"] == "dart":
                params["max_depth"] = trial.suggest_int(
                    "max_depth", optuna_config.max_depth[0], optuna_config.max_depth[1]
                )

            params["seed"] = self.random_seed
            early_stopping_rounds: int = params.pop("early_stopping_rounds", 10)  # type: ignore
            num_boost_round: int = params.pop("num_boost_round", 500)  # type: ignore

            # Prepare data
            data_dict: dict[str, Any] = self.data_dict
            x_train, y_train = data_dict["x_train"], data_dict["y_train"]
            x_val, y_val = data_dict["x_val"], data_dict["y_val"]
            dtrain = xgb.DMatrix(x_train, y_train, enable_categorical=True)
            dval = xgb.DMatrix(x_val, y_val, enable_categorical=True)

            # Train the model
            model: xgb.Booster = xgb.train(
                params=params,
                dtrain=dtrain,
                num_boost_round=num_boost_round,
                verbose_eval=100,
                evals=[(dval, "validation")],
                early_stopping_rounds=early_stopping_rounds,
            )
            preds = model.predict(dval)
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

            # Log to MLflow
            self.mlflow_tracker.log_params(params)
            self.mlflow_tracker.log_metrics(  # type: ignore
                {
                    "RMSE": mean_rmse,
                    "MAE": mean_mae,
                    "MAPE": mean_mape,
                    "Adjusted_R2": mean_adj_r2,
                }
            )

        return mean_rmse  # type: ignore

    def _objective_lightgbm(self, trial: optuna.Trial) -> float:
        with self.mlflow_tracker.start_run(nested=True):
            optuna_config = app_config.optuna_config.lightgbm_optuna_config
            # Define hyperparameters
            # NB: Use `log=True` for parameters that span several orders of magnitude
            # e.g , learning_rate, reg_alpha, reg_lambda, etc.
            params = {
                "objective": optuna_config.objective,
                "metric": optuna_config.metric,
                "reg_lambda": trial.suggest_float(
                    "reg_lambda",
                    optuna_config.reg_lambda[0],
                    optuna_config.reg_lambda[1],
                    log=True,
                ),
                "reg_alpha": trial.suggest_float(
                    "reg_alpha",
                    optuna_config.reg_alpha[0],
                    optuna_config.reg_alpha[1],
                    log=True,
                ),
                "learning_rate": trial.suggest_float(
                    "learning_rate",
                    optuna_config.learning_rate[0],
                    optuna_config.learning_rate[1],
                    log=True,
                ),
                "num_leaves": trial.suggest_int(
                    "num_leaves",
                    optuna_config.num_leaves[0],
                    optuna_config.num_leaves[1],
                ),
                "max_depth": trial.suggest_int(
                    "max_depth", optuna_config.max_depth[0], optuna_config.max_depth[1]
                ),
                "min_child_samples": trial.suggest_int(
                    "min_child_samples",
                    optuna_config.min_child_samples[0],
                    optuna_config.min_child_samples[1],
                ),
                "verbosity": -1,  # Suppress warnings
                "seed": self.random_seed,
            }

            # Training control params
            num_boost_round: int = trial.suggest_int(
                "num_boost_round",
                optuna_config.num_boost_round[0],
                optuna_config.num_boost_round[1],
            )
            early_stopping_rounds: int = trial.suggest_int(
                "early_stopping_rounds",
                optuna_config.early_stopping_rounds[0],
                optuna_config.early_stopping_rounds[1],
            )

            data_dict: dict[str, Any] = self.data_dict

            x_train, y_train = data_dict["x_train"], data_dict["y_train"]
            x_val, y_val = data_dict["x_val"], data_dict["y_val"]

            # Create datasets for LightGBM
            lgb_train = lgb.Dataset(x_train, y_train)
            lgb_val = lgb.Dataset(x_val, y_val, reference=lgb_train)

            # Train with proper early stopping
            model = lgb.train(
                params,
                lgb_train,
                num_boost_round=num_boost_round,
                valid_sets=[lgb_val],
                valid_names=["validation"],
                callbacks=[
                    lgb.early_stopping(
                        stopping_rounds=early_stopping_rounds, verbose=False
                    ),
                    lgb.log_evaluation(period=0),  # Suppress per-iteration logs
                ],
            )
            preds = model.predict(x_val, num_iteration=model.best_iteration)
            metrics: dict[str, float | None] = compute_metrics(
                y_val,
                preds,  # type: ignore
                n_features=x_train.shape[1],
            )  # type: ignore

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

            # Log to MLflow
            self.mlflow_tracker.log_params(params)
            self.mlflow_tracker.log_metrics(metrics)  # type: ignore

        return mean_rmse  # type: ignore

    def _hyperparameter_tuning_random_forest(self) -> dict[str, Any]:
        """
        Perform hyperparameter tuning for a RandomForestRegressor using Optuna and log results to MLflow.
        """
        optuna_config = app_config.optuna_config.random_forest_optuna_config
        n_trials: int = optuna_config.n_trials

        logger.info(
            f"🚨 Starting hyperparameter tuning for Random Forest with {n_trials} trials..."
        )
        with self.mlflow_tracker.start_run(nested=True) as run_id:
            study = optuna.create_study(
                direction="minimize",
                sampler=optuna.samplers.TPESampler(seed=self.random_seed),
            )
            study.optimize(
                self._objective_random_forest,
                n_trials=n_trials,
                callbacks=[self.champion_callback],
            )
            best_trial = study.best_trial
            best_params: dict[str, Any] = best_trial.params
            rmse: float = best_trial.user_attrs.get("RMSE", 0.0)
            mae: float = best_trial.user_attrs.get("MAE", 0.0)
            mape: float = best_trial.user_attrs.get("MAPE", 0.0)
            adj_r2: float | None = best_trial.user_attrs.get("Adjusted_R2", None)
            print(f"Best trial: {best_trial.number} | Value: {best_trial.value}")

            # Log best parameters and metrics to MLflow
            self.mlflow_tracker.log_params(best_params)
            self.mlflow_tracker.log_metrics(
                {
                    "best_rmse": best_trial.value,
                    "mean_rmse": rmse,
                    "mean_mae": mae,
                    "mean_mape": mape,
                    "mean_adjusted_r2": adj_r2 if adj_r2 is not None else None,
                }
            )
            # Log tags
            tags: dict[str, Any] = (
                app_config.experiment_config.experiment_tags.model_dump()
            )
            model_name = ModelType.RANDOM_FOREST
            tags["model_family"] = model_name
            self.mlflow_tracker.set_tags(tags=tags)

            # Build model
            tscv = TimeSeriesSplit(
                n_splits=self.n_splits,
                test_size=self.cv_test_size,
                gap=0,
            )
            rf_model = RandomForestRegressor(
                **best_params, random_state=self.random_seed
            )
            data_dict: dict[str, Any] = self.data_dict
            x_train, y_train = data_dict["x_train"], data_dict["y_train"]
            cv_results: dict[str, Any] = cross_validate_sklearn_model(
                tscv, x_train, y_train, rf_model
            )
            trained_model: RandomForestRegressor = cv_results.get("model")  # type: ignore

            # Feature importance
            column_names = data_dict["columns"]
            weights_ = trained_model.feature_importances_

            feat_importance: dict[str, Any]
            (_, feat_importance) = get_model_feature_importance(
                model_name=model_name, features=column_names, weights=weights_, n=20
            )
            self.mlflow_tracker.log_mlflow_artifact(
                object=feat_importance,
                object_type=ArtifactsType.JSON,
                filename="feat_imp_random_forest",
                artifact_dest="models/",
            )
            self.mlflow_tracker.log_model(
                trained_model,
                model_name=model_name,
                input_example=self.input_example,
            )
            model_uri: str = self.mlflow_tracker.get_artifact_uri(
                artifact_path="models/"
            )

        return {
            "run_id": run_id,
            "model_name": model_name,
            "best_params": best_params,
            "model_uri": model_uri,
        }

    def _hyperparameter_tuning_xgboost(self) -> dict[str, Any]:
        optuna_config = app_config.optuna_config.xgboost_optuna_config
        n_trials: int = optuna_config.n_trials

        logger.info(
            f"🚨 Starting hyperparameter tuning for XGBoost with {n_trials} trials..."
        )
        with self.mlflow_tracker.start_run(nested=True) as run_id:
            study = optuna.create_study(
                direction="minimize",
                sampler=optuna.samplers.TPESampler(seed=self.random_seed),
            )
            study.optimize(
                self._objective_xgboost,
                n_trials=n_trials,
                callbacks=[self.champion_callback],
            )
            best_trial = study.best_trial
            best_params: dict[str, Any] = best_trial.params
            print(f"Best trial: {best_trial.number} | Value: {best_trial.value}")

            # Log tags
            tags: dict[str, Any] = (
                app_config.experiment_config.experiment_tags.model_dump()
            )
            model_name = ModelType.XGBOOST
            tags["model_family"] = model_name
            self.mlflow_tracker.set_tags(tags=tags)

            # Build model
            xgb_result: dict[str, Any] = self._train_xgboost(best_params)
            xgb_model = xgb_result.get("model")
            rmse: float | None = xgb_result["metrics"].get("RMSE", None)  # type: ignore
            mae: float | None = xgb_result["metrics"].get("MAE", None)  # type: ignore
            mape: float | None = xgb_result["metrics"].get("MAPE", None)  # type: ignore
            adj_r2: float | None = xgb_result["metrics"].get("Adjusted_R2", None)  # type: ignore

            # Feature importance
            column_names = self.data_dict["columns"]
            weights_dict: dict[str, float] = get_feature_importance_from_booster(
                xgb_model,
                column_names,
                importance_type="weight",
            )
            weights_ = list(weights_dict.values())
            (_, feat_importance) = get_model_feature_importance(
                model_name=model_name, features=column_names, weights=weights_, n=20
            )
            # Log best parameters and metrics to MLflow
            self.mlflow_tracker.log_params(best_params)
            self.mlflow_tracker.log_metrics(
                {
                    "best_rmse": best_trial.value,
                    "mean_rmse": rmse,
                    "mean_mae": mae,
                    "mean_mape": mape,
                    "mean_adjusted_r2": adj_r2 if adj_r2 is not None else None,
                }
            )
            self.mlflow_tracker.log_mlflow_artifact(
                object=feat_importance,
                object_type=ArtifactsType.JSON,
                filename="feat_imp_xgboost",
                artifact_dest="models/",
            )
            self.mlflow_tracker.log_model(
                xgb_model,
                model_name=model_name,
                input_example=self.input_example,
                save_format=ArtifactsType.JSON,
            )
            model_uri: str = self.mlflow_tracker.get_artifact_uri(
                artifact_path="models/"
            )

        return {
            "run_id": run_id,
            "model_name": model_name,
            "best_params": best_params,
            "model_uri": model_uri,
        }

    def _hyperparameter_tuning_lightgbm(self) -> dict[str, Any]:
        optuna_config = app_config.optuna_config.lightgbm_optuna_config
        n_trials: int = optuna_config.n_trials

        logger.info(
            f"🚨 Starting hyperparameter tuning for LightGBM with {n_trials} trials..."
        )
        with self.mlflow_tracker.start_run(nested=True) as run_id:
            study = optuna.create_study(
                direction="minimize",
                sampler=optuna.samplers.TPESampler(seed=self.random_seed),
            )
            study.optimize(
                self._objective_lightgbm,
                n_trials=n_trials,
                callbacks=[self.champion_callback],
            )
            best_trial = study.best_trial
            best_params: dict[str, Any] = best_trial.params
            print(f"Best trial: {best_trial.number} | Value: {best_trial.value}")

            # Log tags
            tags: dict[str, Any] = (
                app_config.experiment_config.experiment_tags.model_dump()
            )
            model_name = ModelType.LIGHTGBM
            tags["model_family"] = model_name
            self.mlflow_tracker.set_tags(tags=tags)

            # Build final model with best params
            lgb_result: dict[str, Any] = self._train_lightgbm(best_params)
            lgb_model = lgb_result.get("model")
            rmse: float | None = lgb_result["metrics"].get("RMSE", None)  # type: ignore
            mae: float | None = lgb_result["metrics"].get("MAE", None)  # type: ignore
            mape: float | None = lgb_result["metrics"].get("MAPE", None)  # type: ignore
            adj_r2: float | None = lgb_result["metrics"].get("Adjusted_R2", None)  # type: ignore

            # Feature importance
            column_names = self.data_dict["columns"]
            feat_imp_df: pl.DataFrame = get_lightgbm_feature_importance(
                lgb_model, column_names
            )
            weights_ = feat_imp_df["gain_importance_normalized"].to_list()

            (_, feat_importance) = get_model_feature_importance(
                model_name=model_name, features=column_names, weights=weights_, n=20
            )
            # Log best parameters and metrics to MLflow
            self.mlflow_tracker.log_params(best_params)
            self.mlflow_tracker.log_metrics(
                {
                    "best_rmse": best_trial.value,
                    "mean_rmse": rmse,
                    "mean_mae": mae,
                    "mean_mape": mape,
                    "mean_adjusted_r2": adj_r2 if adj_r2 is not None else None,
                }
            )
            self.mlflow_tracker.log_mlflow_artifact(
                object=feat_importance,
                object_type=ArtifactsType.JSON,
                filename="feat_imp_lightgbm",
                artifact_dest="models/",
            )
            self.mlflow_tracker.log_model(
                lgb_model,
                model_name=model_name,
                input_example=self.input_example,
                save_format=ArtifactsType.TXT,
            )
            model_uri: str = self.mlflow_tracker.get_artifact_uri(
                artifact_path="models/"
            )

        return {
            "run_id": run_id,
            "model_name": model_name,
            "best_params": best_params,
            "model_uri": model_uri,
        }

    def hyperparameter_tuning_all_models(self) -> list[dict[str, Any]]:
        """Perform hyperparameter tuning for all models and log results to MLflow."""
        results: list[dict[str, Any]] = []

        try:
            logger.info("🚨 Starting hyperparameter tuning for all models...")
            # ================================
            # ========= Random Forest ========
            # ================================
            rf_results: dict[str, Any] = self._hyperparameter_tuning_random_forest()
            results.append(rf_results)

            # ================================
            # ============ XGBoost ===========
            # ================================
            xgb_results: dict[str, Any] = self._hyperparameter_tuning_xgboost()
            results.append(xgb_results)

            # ================================
            # ============ LightGBM ==========
            # ================================
            lgb_results: dict[str, Any] = self._hyperparameter_tuning_lightgbm()
            results.append(lgb_results)

            # All done
            logger.info("✅ Hyperparameter tuning completed successfully.")

            return results

        except HyperparameterTuningError as e:
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

    @staticmethod
    def _format_metrics(metrics: dict[str, float], prefix: str) -> dict[str, float]:
        """Format metric names by adding a prefix."""
        return {f"{prefix}_{k}": v for k, v in metrics.items() if v is not None}
