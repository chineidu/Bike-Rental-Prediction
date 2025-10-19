from typing import Any

import numpy as np
import polars as pl

from src.api.utilities.model_loader.loader import Data
from src.config import app_config
from src.ml.feature_engineering import FeatureEngineer
from src.schemas.types import CurrencyType, PredictionResultDict, WeatherDict


def convert_to_original_temp(
    t_norm: float, t_min: float = -8, t_max: float = 39
) -> float:
    """Convert normalized temperature back to original scale (Celsius)."""
    return t_norm * (t_max - t_min) + t_min


def calculate_temp_factor(temp: float) -> int:
    """Calculate temperature factor based on given temperature in celsius."""
    temp = np.round(temp, 2)
    if temp < 0 or temp > 35:
        return 4
    if 0 <= temp <= 9 or 30 <= temp <= 35:
        return 3
    if 21 <= temp <= 29:
        return 2
    if 10 <= temp <= 20:
        return 1
    return 4  # Default case (should not occur)


def calculate_hum_factor(hum: float) -> int:
    """Calculate humidity factor based on given humidity (0 to 1)."""
    hum = np.round(hum, 2)
    if hum > 0.8:
        return 4
    if hum < 0.2:
        return 3
    if 0.2 <= hum <= 0.39 or 0.61 <= hum <= 0.8:
        return 2
    if 0.4 <= hum <= 0.6:
        return 1
    return 4


def convert_to_original_windspeed(w_norm: float, w_factor: float = 0.8507) -> float:
    """Convert normalized windspeed back to original scale (km/h)."""
    return w_norm * w_factor


def calculate_windspeed_factor(windspeed: float) -> int:
    """
    Calculate a discrete windspeed factor used for dynamic pricing based on wind speed.

    Parameters
    ----------
    windspeed : float
        Wind speed in kilometers per hour (km/h). The value is rounded to 2 decimal
        places before evaluating which factor to return.

    Returns
    -------
    int
    """
    windspeed = np.round(windspeed, 2)
    if windspeed > 50:
        return 4
    if 26 <= windspeed <= 50:
        return 3
    if 16 <= windspeed <= 25:
        return 2
    if 0 <= windspeed <= 15:
        return 1
    return 1


def calculate_weather_factor(
    temp: int, hum: int, windspeed: int, weathersit: int
) -> float:
    """
    Compute a combined weather factor from temperature, humidity, wind speed, and a weather situation code.

    Lower values indicate better weather conditions for bike rentals.

    Parameters
    ----------
    temp : int
        Temperature value (integer). Higher values increase the contribution to the factor.
    hum : int
        Humidity value (integer). Higher values increase the contribution to the factor.
    windspeed : int
        Wind speed value (integer). Higher values increase the contribution to the factor.
    weathersit : int
        Encoded weather situation (integer). Represents categorical/severity of weather (e.g., clear=0,
        light rain=1, heavy rain=2); larger values increase the scaling effect.

    Returns
    -------
    float
        Weather factor where lower is better. The result is clipped to the interval [0.85, 1.7] and
        rounded to two decimal places.
    """
    min_value, max_value = 0.85, 1.48
    min_allowed, max_allowed = 1, 4

    if not all(
        isinstance(x, int) and x >= 0 for x in [temp, hum, windspeed, weathersit]
    ):
        raise ValueError("All inputs must be non-negative integers.")
    # Must be between 1 and 4
    if not (min_allowed <= temp <= max_allowed):
        raise ValueError("temp must be between 1 and 4.")
    if not (min_allowed <= hum <= max_allowed):
        raise ValueError("hum must be between 1 and 4.")
    if not (min_allowed <= windspeed <= max_allowed):
        raise ValueError("windspeed must be between 1 and 4.")
    if not (min_allowed <= weathersit <= max_allowed):
        raise ValueError("weathersit must be between 1 and 4.")

    base_factor: float = 0.4
    weights: WeatherDict = WeatherDict(
        weathersit=0.33, temp=0.25, windspeed=0.16, hum=0.1
    )

    # Temperature
    base_factor += weights["temp"] * temp
    # Humidity
    base_factor += weights["hum"] * hum
    # Windspeed
    base_factor += weights["windspeed"] * windspeed
    # Weather Situation
    base_factor *= weights["weathersit"] * weathersit

    # Clip between 0.85 and 1.7
    return round(max(min_value, min(base_factor, max_value)), 2)


def calculate_dynamic_elasticity(
    base_elasticity: float, utilization_rate: float, weather_factor: float = 1.0
) -> float:
    """
    Calculate a dynamic price elasticity adjusted for utilization and weather.

    This function adjusts a base elasticity value by applying a utilization-dependent
    modifier and a multiplicative weather factor. Sensitivity to utilization is
    controlled by an internal coefficient alpha (0.6). The computed elasticity is
    rounded to two decimal places before being returned.

    Parameters
    ----------
    base_elasticity : float
        The baseline elasticity value to be adjusted (can be positive or negative).
    utilization_rate : float
        Current utilization rate, expected on a 0.0 - 1.0 scale where 0.0 indicates no
        utilization and 1.0 indicates full utilization.
    weather_factor : float, optional
        Multiplicative factor representing weather influence on elasticity
        (default is 1.0). Values >1 amplify the elasticity, values <1 reduce it.

    Returns
    -------
    float
        The adjusted elasticity value, rounded to two decimal places.
    """
    # Controls sensitivity to utilization rate
    # Adjust elasticity based on utilization and weather
    alpha = 0.6

    if not (0.0 <= utilization_rate <= 1.0):
        raise ValueError("utilization_rate must be between 0.0 and 1.0")

    elasticity = base_elasticity * (alpha * weather_factor + (1 - utilization_rate))
    return round(elasticity, 2)


def calculate_time_factor(time: int) -> float:
    """Calculate time factor based on hour of the day (0-23)."""
    peak_hr: set[int] = {7, 8, 9, 16, 17, 18}
    business_hr: set[int] = {10, 11, 12, 13, 14, 15, 19, 20}

    if not 0 <= time <= 23:
        raise ValueError("Hour must be between 0 and 23")
    if time in peak_hr:
        return 1.48
    if time in business_hr:
        return 1.30
    return 0.92


def calculate_competitor_factor(base_price: float, competitor_price: float) -> float:
    """Calculate a competitor adjustment factor for dynamic pricing.

    Parameters
    ----------
    base_price : float
        The reference/base price to compare against. Must be non-zero.
    competitor_price : float
        The competitor's price. Must be greater than or equal to zero.

    Returns
    -------
    float
        A clipped multiplicative factor.
    """
    min_value, max_value = (0.85, 1.5)
    alpha: float = 0.45

    if competitor_price < 0:
        raise ValueError("competitor_price cannot be a negative value.")

    if base_price == 0:
        raise ValueError("base_price must be non-zero to compute percentage change")

    pct_change: float = (base_price - competitor_price) / base_price
    result: float = round(1 - (alpha * pct_change), 2)

    return np.clip(result, a_min=min_value, a_max=max_value).item()


def calculate_price_multiplier(
    base_price: float,
    competitor_price: float,
    demand: float,
    weather_factor: float,
    time: int,
    base_elasticity: float = -1.1,
) -> float:
    """
    Calculate the price multiplier for dynamic pricing.

    Parameters
    ----------
    base_price : float
        The current base price of the item or service.
    competitor_price : float
        The competitor's price for the same or comparable item/service. Used to compute
        a competitor adjustment factor.
    demand : float
        Current demand for the service (e.g., fraction or percentage expressed as 0-1).
        Higher demand increases the surge component.
    weather_factor : float
        A multiplicative factor capturing weather-driven demand effects (e.g., >1 for
        favorable demand-increasing conditions).
    time : int
        Time indicator used to compute a time-of-day factor (e.g., hour of day 0-23).
    base_elasticity : float, optional
        Base price elasticity of demand (typically negative). Default is -1.1.

    Returns
    -------
    float
        The computed price multiplier.
    """
    # Tunable
    k: float = 0.662
    capacity: float = app_config.business_config.max_capacity
    inventory: float = capacity - demand

    if inventory < 0:
        raise ValueError("Demand cannot exceed capacity")
    utilization_rate = (capacity - inventory) / capacity

    surge: float = 1 + (k * utilization_rate)
    comp_factor: float = calculate_competitor_factor(base_price, competitor_price)
    dyn_elasticity: float = calculate_dynamic_elasticity(
        base_elasticity, utilization_rate, weather_factor
    )
    time_factor: float = calculate_time_factor(time)
    price_multiplier: float = (
        k * comp_factor * surge * time_factor * (1 / np.abs(dyn_elasticity))
    )

    return round(price_multiplier, 2)


def calculate_price(
    base_price: float, price_multiplier: float, currency: str
) -> dict[str, Any]:
    """
    Calculate the price based on a base price and a multiplier.

    Parameters
    ----------
    base_price : float
        Base price used as the starting point for the calculation. Must be non-negative.
    price_multiplier : float
        Multiplier applied to the base price. Must be greater than zero.

    Returns
    -------
    dict[str, Any]

    """
    min_price, max_price = (
        app_config.business_config.min_price,
        app_config.business_config.max_price,
    )

    if base_price < 0:
        raise ValueError("Base price cannot be negative")
    if price_multiplier <= 0:
        raise ValueError("Price multiplier must be greater than zero")

    final_price: float = base_price * price_multiplier
    final_price = np.clip(
        round(final_price, 2), a_min=min_price, a_max=max_price
    ).item()
    other_factor: float = round(final_price - base_price, 2)

    return {
        "base_price": base_price,
        "price": final_price,
        "price_multiplier": price_multiplier,
        "surge": other_factor if other_factor > 0 else 0.0,
        "discount": other_factor if other_factor < 0 else 0.0,
        "min_price": min_price,
        "max_price": max_price,
        "currency": currency,
    }
    # return PredictionResultDict(
    #     base_price=base_price,
    #     price=final_price,
    #     price_multiplier=price_multiplier,
    #     surge=other_factor if other_factor > 0 else 0.0,
    #     discount=other_factor if other_factor < 0 else 0.0,
    #     min_price=min_price,
    #     max_price=max_price,
    #     currency=currency,
    # )


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


def predict_user_demand(model: Any, features: Data) -> int:
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


def calculate_final_price(
    data: Data,
    base_price: float,
    base_elasticity: float,
    demand: float,
    competitor_price: float,
    currency: str,
) -> PredictionResultDict:
    """Make a price prediction using the provided model and features."""

    if currency == CurrencyType.NGN:
        pass  # No conversion needed for NGN

    # Convert features to DataFrame
    raw_df: pl.DataFrame = pl.from_records(data)
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
    _pred: dict[str, Any] = calculate_price(
        base_price=base_price, price_multiplier=price_multiplier, currency=currency
    )
    pred: PredictionResultDict = PredictionResultDict(**_pred)  # type: ignore
    pred["demand"] = demand
    pred["competitor_price"] = competitor_price
    return pred


def get_competitor_price(data: Data) -> float:
    """Get competitor price based on data."""
    min_price: float = app_config.business_config.min_competitor_price
    max_price: float = app_config.business_config.max_competitor_price

    comp_predictor = CompetitorPricePredictor(min_price=min_price, max_price=max_price)
    return comp_predictor.predict_price(data[0])


class CompetitorPricePredictor:
    """Rule-based algorithm to predict bike rental prices on an hourly basis using Polars."""

    def __init__(
        self,
        min_price: float = 500.0,
        max_price: float = 1000.0,
        weights: dict[str, float] | None = None,
    ) -> None:
        """Initialize the price predictor.

        Parameters
        ----------
        min_price : float, optional
            Minimum price in the range, by default 500.0
        max_price : float, optional
            Maximum price in the range, by default 1000.0
        weights : dict[str, float] | None, optional
            Feature weights for price calculation, by default None
        """
        self.min_price: float = min_price
        self.max_price: float = max_price
        self.price_range: float = max_price - min_price

        # Default weights
        self.weights: dict[str, float] = weights or {
            "mnth": 0.04,
            "holiday": 0.05,
            "hr": 0.15,
            "weekday": 0.05,
            "workingday": 0.1,
            "weathersit": 0.12,
            "temp": 0.06,
            "hum": 0.02,
            "windspeed": 0.02,
            "cnt": 0.16,
        }

    def _normalize_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """Normalize features to [0, 1] range.

        Parameters
        ----------
        df : pl.DataFrame
            DataFrame with raw feature values

        Returns
        -------
        pl.DataFrame
            DataFrame with normalized feature values
        """
        return df.with_columns(
            [
                # Month: 1-12 -> [0, 1]
                ((pl.col("mnth").fill_null(6) - 1) / 11).alias("mnth_norm"),
                # Holiday: 0 or 1 (already normalized)
                pl.col("holiday").fill_null(0).cast(pl.Float64).alias("holiday_norm"),
                # Hour: 0-23 -> [0, 1], with peak hours weighted higher
                pl.when(pl.col("hr").fill_null(12).is_in([7, 8, 9, 17, 18, 19]))
                .then(0.8 + (0.2 * (pl.col("hr") % 24) / 23))
                .otherwise((pl.col("hr") % 24) / 23)
                .alias("hr_norm"),
                # Weekday: 0-6 -> [0, 1]
                (pl.col("weekday").fill_null(3) / 6).alias("weekday_norm"),
                # Working day: 0 or 1 (already normalized)
                pl.col("workingday")
                .fill_null(1)
                .cast(pl.Float64)
                .alias("workingday_norm"),
                # Weather situation: 1-4 -> [0, 1] (inverted, since 1=best, 4=worst)
                (1 - ((pl.col("weathersit").fill_null(1) - 1) / 3)).alias(
                    "weathersit_norm"
                ),
                # Temperature: 0-1 with penalty for non-optimal temps
                pl.when(
                    (pl.col("temp").fill_null(0.5) >= 0.6) & (pl.col("temp") <= 0.8)
                )
                .then(pl.col("temp"))
                .otherwise(pl.col("temp") * 0.8)
                .alias("temp_norm"),
                # Humidity: 0-1 (inverted, lower humidity is better)
                (1 - pl.col("hum").fill_null(0.5)).alias("hum_norm"),
                # Wind speed: 0-1 (inverted, lower wind is better)
                (1 - pl.col("windspeed").fill_null(0.2)).alias("windspeed_norm"),
                # Count (demand): normalize based on expected range (0-1000)
                pl.min_horizontal(pl.col("cnt").fill_null(200) / 1000, 1.0).alias(
                    "cnt_norm"
                ),
            ]
        )

    def _calculate_base_score(self, df: pl.DataFrame) -> pl.DataFrame:
        """Calculate weighted score from normalized features.

        Parameters
        ----------
        df : pl.DataFrame
            DataFrame with normalized feature values

        Returns
        -------
        pl.DataFrame
            DataFrame with added 'base_score' column
        """
        # Calculate weighted sum of normalized features
        score_expr: pl.Expr = pl.lit(0.0)
        for feature, weight in self.weights.items():
            score_expr = score_expr + (pl.col(f"{feature}_norm") * weight)

        return df.with_columns(score_expr.alias("base_score"))

    def predict_price(self, features: dict[str, Any]) -> float:
        """Predict the hourly bike rental price for a single instance.

        Parameters
        ----------
        features : dict[str, Any]
            Dictionary containing feature values

        Returns
        -------
        float
            Predicted price in the range [min_price, max_price]
        """
        # Convert single dict to DataFrame
        df = pl.DataFrame([features])

        # Get predictions
        result: pl.DataFrame = self.predict_dataframe(df)

        return result["predicted_price"][0]

    def predict_batch(self, features_list: list[dict[str, Any]]) -> list[float]:
        """Predict prices for multiple feature sets.

        Parameters
        ----------
        features_list : list[dict[str, Any]]
            List of feature dictionaries

        Returns
        -------
        list[float]
            List of predicted prices
        """

        # Convert list of dicts to DataFrame
        df = pl.DataFrame(features_list)

        # Get predictions
        result: pl.DataFrame = self.predict_dataframe(df)

        return result["predicted_price"].to_list()

    def predict_dataframe(self, df: pl.DataFrame) -> pl.DataFrame:
        """Predict prices for a DataFrame of features.

        Parameters
        ----------
        df : pl.DataFrame
            DataFrame containing feature columns:
            - mnth: Month (1-12)
            - holiday: Holiday flag (0 or 1)
            - hr: Hour (0-23)
            - weekday: Day of week (0-6)
            - workingday: Working day flag (0 or 1)
            - weathersit: Weather situation (1-4, where 1=best, 4=worst)
            - temp: Normalized temperature (0-1)
            - hum: Normalized humidity (0-1)
            - windspeed: Normalized wind speed (0-1)
            - cnt: Rental count/demand (0-1000+)

        Returns
        -------
        pl.DataFrame
            Original DataFrame with added 'predicted_price' column
        """
        columns: list[str] = [
            "mnth",
            "holiday",
            "hr",
            "weekday",
            "workingday",
            "weathersit",
            "temp",
            "hum",
            "windspeed",
            "cnt",
        ]
        # Ensure only relevant columns are used
        df = df.select(columns)
        # Normalize features
        df_normalized: pl.DataFrame = self._normalize_features(df)

        # Calculate base score
        df_scored: pl.DataFrame = self._calculate_base_score(df_normalized)

        # Map score to price range and ensure bounds
        return df_scored.with_columns(
            [
                (self.min_price + (pl.col("base_score") * self.price_range))
                .clip(self.min_price, self.max_price)
                .round(2)
                .alias("predicted_price")
            ]
        )
