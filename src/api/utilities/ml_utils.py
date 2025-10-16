from typing import Any

import numpy as np
import polars as pl

from src.config import app_config
from src.ml.dynamic_pricing import (
    CompetitorPricePredictor,
    calculate_hum_factor,
    calculate_price,
    calculate_price_multiplier,
    calculate_temp_factor,
    calculate_weather_factor,
    calculate_windspeed_factor,
    convert_to_original_temp,
    convert_to_original_windspeed,
)
from src.ml.feature_engineering import FeatureEngineer
from src.schemas import PredictedPriceResponse


def format_weather_features(weather: dict[str, Any]) -> dict[str, Any]:
    """Format and transform weather features."""
    columns = ["temp", "hum", "windspeed", "weathersit"]
    df: pl.DataFrame = pl.DataFrame([weather]).select(columns)
    df = df.with_columns(
        pl.col("temp")
        .map_batches(lambda x: convert_to_original_temp(x), return_dtype=pl.Float64)
        .alias("temp"),
        pl.col("windspeed")
        .map_elements(
            lambda x: convert_to_original_windspeed(x), return_dtype=pl.Float64
        )
        .alias("windspeed"),
    ).with_columns(
        pl.col("temp")
        .map_elements(lambda x: calculate_temp_factor(x), return_dtype=pl.Int8)
        .alias("temp"),
        pl.col("hum")
        .map_elements(lambda x: calculate_hum_factor(x), return_dtype=pl.Int8)
        .alias("hum"),
        pl.col("windspeed")
        .map_elements(lambda x: calculate_windspeed_factor(x), return_dtype=pl.Int8)
        .alias("windspeed"),
    )
    return df.to_dicts()[0]


def predict_user_demand(model: Any, features: list[dict[str, Any]]) -> int:
    """Make a demand prediction using the provided model and features."""
    default_value: float = 0.0
    target_col: str = "target"

    # Convert features to DataFrame and apply feature engineering
    feature_df: pl.DataFrame = pl.from_records(features).with_columns(
        # Add missing columns with default values
        pl.lit(default_value).alias("atemp"),
        pl.lit(default_value).alias("casual"),
        pl.lit(default_value).alias("registered"),
    )
    feat_eng = FeatureEngineer()
    feature_df = feat_eng.transform(
        data=feature_df, config=app_config.feature_config
    ).drop(target_col)

    # Make prediction
    prediction = model.predict(feature_df)

    # Ensure the prediction is a non-negative integer
    return np.clip(int(prediction[0]), a_min=0, a_max=None).item()


def estimate_price(
    features: list[dict[str, Any]],
    base_price: float,
    base_elasticity: float,
    demand: float,
    competitor_price: float,
) -> PredictedPriceResponse:
    """Make a price prediction using the provided model and features."""

    # Convert features to DataFrame
    raw_df: pl.DataFrame = pl.from_records(features)
    # Prepare weather features
    weather_features: dict[str, Any] = format_weather_features(raw_df.to_dicts()[0])
    weather_factor: float = calculate_weather_factor(**weather_features)
    time: int = int(raw_df["hr"][0])
    price_multiplier: float = calculate_price_multiplier(
        base_price=base_price,
        competitor_price=competitor_price,
        demand=demand,
        weather_factor=weather_factor,
        time=time,
        base_elasticity=base_elasticity,
    )
    pred: PredictedPriceResponse = calculate_price(
        base_price=base_price, price_multiplier=price_multiplier
    )
    return pred


def get_competitor_price(features: dict[str, Any]) -> float:
    """Get competitor price based on features."""
    min_price: float = app_config.business_config.min_competitor_price
    max_price: float = app_config.business_config.max_competitor_price

    comp_predictor = CompetitorPricePredictor(min_price=min_price, max_price=max_price)
    return comp_predictor.predict_price(features)


if __name__ == "__main__":
    sample_weather = {
        "temp": 0.5,
        "hum": 0.76,
        "windspeed": 0.1,
        "weather_sit": 1,
    }
    formatted_weather = format_weather_features(sample_weather)
    print(formatted_weather)

weights = {
    "mnth": 0.1,
    "holiday": 0.05,
    "hr": 0.25,
    "weekday": 0.05,
    "workingday": 0.06,
    "weathersit": 0.15,
    "temp": 0.1,
    "hum": 0.02,
    "windspeed": 0.02,
    "cnt": 0.2,
}
