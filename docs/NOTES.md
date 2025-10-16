# Additional Notes

## Elasticity

- This is the responsiveness of demand to changes in price.

- Formula:

$$ Elasticity = \frac{\% \Delta \text{Quantity Demanded}}{\% \Delta \text{Price}} $$

- For example, a 10% increase in rental price might lead to a 2% decrease in bike rentals, indicating an elasticity of -0.2.

$$ Elasticity = \frac{-2\%}{10\%} = -0.2 $$

- In relation to the bike rental data, elasticity can help us understand how changes in factors like temperature, weather conditions, and rental prices impact overall demand.

### Elasticity Interpretation

- Elasticity > 1: Demand is elastic (sensitive to price changes)
- Elasticity < 1: Demand is inelastic (less sensitive to price changes)
- Elasticity = 1: Unit elastic (proportional response to price changes)

## Weather Impact Factors

- Weather conditions have a significant effect on bike rental demand, which can be quantified using impact factors assigned to different weather scenarios.

- Severe weather (heavy rain, snow) reduces rentals more than mild weather conditions.

- Lower impact factors indicate favorable weather that boosts rentals, while higher factors represent adverse conditions that suppress demand.

  - Overall Impact Factors:
    - Good weather (Clear, Few clouds): < 0.8
    - Moderate weather (Scattered clouds, Broken clouds): 0.8 - 1.2
    - Poor weather (Shower rain, Rain, Thunderstorm): > 1.2
    - Severe weather (Snow, Mist): > 1.5

```py
def convert_to_original_temp(t_norm: float, t_min: float = -8, t_max: float = 39) -> float:
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
    """Calculate windspeed factor based on given windspeed (km/h)."""
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

from typing import TypedDict


class WeatherDict(TypedDict):
    temp: float
    hum: float
    windspeed: float
    weather_sit: float


def normalize_factor(value: int) -> float:
    min_value, max_value = (1, 4)

    if not min_value <= value <= max_value:
        raise (f"{value} should be between 1 and 4")

    norm_value: float = (value - min_value) / (max_value - min_value)
    return round(norm_value, 2)


def calculate_weather_factor(
    temp: int, hum: int, windspeed: int, weather_sit: int
) -> float:
    """Lower is better."""
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
    base_elasticity: float, utilization_rate: float, weather_factor=1.0
):
    """Calculate dynamic elasticity based on base elasticity, utilization rate, and weather factor.

    Parameters
    ----------
    base_elasticity : float
        The base elasticity value.
    utilization_rate : float
        The utilization rate, typically between 0 and 1.
    weather_factor : float, optional
        The weather factor, default is 1.0.

    Returns
    -------
    float
        The calculated dynamic elasticity, rounded to 2 decimal places.

    Notes
    -----
    When the weather_factor is high (poor weather), elasticity is high.
    When the utilization rate (demand) is high, elasticity is relatively low.
    """
    # Controls sensitivity to utilization rate
    # Adjust elasticity based on utilization and weather
    alpha = 0.7
    elasticity = base_elasticity * (alpha * weather_factor + (1 - utilization_rate))
    return round(elasticity, 2)

def calculate_time_factor(time: int) -> float:
    """Calculate time factor based on hour of the day (0-23)."""
    peak_hr: set[int] = {7, 8, 9, 16, 17, 18}
    business_hr: set[int] = {10, 11, 12, 13, 14, 15, 19, 20}

    if not 0 <= time <= 23:
        raise ValueError("Hour must be between 0 and 23")
    if time in peak_hr:
        return 1.3
    if time in business_hr:
        return 1.1
    return 0.85

def calculate_competitor_factor(base_price: float, competitor_price: float) -> float:
    min_value, max_value = (0.8, 1.5)
    if competitor_price < 0:
        raise (f"{competitor_price} cannot be a negative value.")

    pct_change = (base_price - competitor_price) / base_price
    return np.clip(1 - pct_change, min=min_value, max=max_value).item()
```
