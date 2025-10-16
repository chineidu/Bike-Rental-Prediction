from typing import Any

import numpy as np

from src.config import app_config
from src.schemas.types import WeatherDict


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
    temp: int, hum: int, windspeed: int, weather_sit: int
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
    weather_sit : int
        Encoded weather situation (integer). Represents categorical/severity of weather (e.g., clear=0,
        light rain=1, heavy rain=2); larger values increase the scaling effect.

    Returns
    -------
    float
        Weather factor where lower is better. The result is clipped to the interval [0.85, 1.7] and
        rounded to two decimal places.
    """
    base_factor: float = 0.5
    weights: WeatherDict = WeatherDict(
        weather_sit=0.4, temp=0.15, windspeed=0.15, hum=0.1
    )

    # Temperature
    base_factor += weights["temp"] * temp
    # Humidity
    base_factor += weights["hum"] * hum
    # Windspeed
    base_factor += weights["windspeed"] * windspeed
    # Weather Situation
    base_factor *= weights["weather_sit"] * weather_sit
    # Clip between 0.85 and 1.7
    return round(max(0.85, min(base_factor, 1.7)), 2)


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
    alpha = 0.7

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
        return 1.5
    if time in business_hr:
        return 1.2
    return 0.85


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
    alpha: float = 0.4

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
    k: float = 0.55
    capacity: float = app_config.business_config.max_capacity
    inventory: float = capacity - demand

    if inventory < 0:
        raise ValueError("Demand cannot exceed capacity")
    utilization_rate = (capacity - inventory) / capacity
    print(f"Utilization Rate: {utilization_rate}")

    surge: float = 1 + (k * utilization_rate)
    print(f"Surge: {surge}")
    comp_factor: float = calculate_competitor_factor(base_price, competitor_price)
    print(f"Competitor Factor: {comp_factor}")
    dyn_elasticity: float = calculate_dynamic_elasticity(
        base_elasticity, utilization_rate, weather_factor
    )
    print(f"Dynamic Elasticity: {dyn_elasticity}")
    time_factor: float = calculate_time_factor(time)
    print(f"Time Factor: {time_factor}")

    price_multiplier: float = (
        k * comp_factor * surge * time_factor * (1 / np.abs(dyn_elasticity))
    )

    return round(price_multiplier, 2)


def calculate_price(base_price: float, price_multiplier: float) -> dict[str, Any]:
    """
    Calculate final price based on a base price and a multiplier.

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
    currency: str = app_config.business_config.currency

    if base_price < 0:
        raise ValueError("Base price cannot be negative")
    if price_multiplier <= 0:
        raise ValueError("Price multiplier must be greater than zero")

    final_price: float = base_price * price_multiplier
    final_price = np.clip(
        round(final_price, 2), a_min=min_price, a_max=max_price
    ).item()

    return {
        "min_price": min_price,
        "max_price": max_price,
        "other_factors": round(final_price - base_price, 2),
        "base_price": base_price,
        "price": final_price,
        "currency": currency,
    }
