#!/usr/bin/env python3
"""
Bike Rental Data Generator

Generates realistic bike rental data with hourly granularity between
specified start and end dates.

Usage:
    python data_gen.py --start "2011-01-01" --end "2011-12-31"
"""

import argparse
import math
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import holidays
import numpy as np
import polars as pl

# Create a single random number generator for consistent seeding
rng = np.random.default_rng(42)

# US holidays for any year (will be dynamically expanded based on date range)
us_holidays = holidays.country_holidays("US")


def get_season(month: int) -> int:
    """Get season based on month (1=spring, 2=summer, 3=fall, 4=winter)"""
    if not (1 <= month <= 12):
        raise ValueError("Month must be between 1 and 12")

    if month in [3, 4, 5]:
        return 1  # spring
    if month in [6, 7, 8]:
        return 2  # summer
    if month in [9, 10, 11]:
        return 3  # fall
    return 4  # winter


def get_weather_situation() -> int:
    """Get weather situation (1=clear, 2=mist, 3=light_rain, 4=heavy_rain)"""
    # Weighted distribution favoring better weather
    return rng.choice([1, 2, 3, 4], p=[0.6, 0.25, 0.13, 0.02])


def calculate_temp_atemp(hour: int, month: int, weather: int) -> tuple[float, float]:
    """Calculate normalized temperature and feeling temperature"""
    # Base temperature varies by month and hour
    month_factor = 0.5 + 0.3 * math.sin((month - 3) * math.pi / 6)  # Peak in summer
    hour_factor = 0.15 * math.sin((hour - 6) * math.pi / 12)  # Peak mid-day
    weather_factor = max(0.1, 1.0 - (weather - 1) * 0.15)  # Lower temp for bad weather

    base_temp = 0.4 + 0.4 * month_factor + hour_factor
    temp = max(0.0, min(1.0, base_temp * weather_factor + rng.normal(0, 0.05)))

    # Feeling temperature is usually close to actual temp but can vary
    atemp = temp + rng.normal(0, 0.03)
    atemp = max(0.0, min(1.0, atemp))

    return round(temp, 4), round(atemp, 4)


def calculate_humidity_windspeed(weather: int) -> tuple[float, float]:
    """Calculate humidity and windspeed based on weather"""
    if weather == 1:  # clear
        hum = rng.uniform(0.3, 0.7)
        windspeed = rng.uniform(0.0, 0.3)
    elif weather == 2:  # mist
        hum = rng.uniform(0.6, 0.9)
        windspeed = rng.uniform(0.1, 0.4)
    elif weather == 3:  # light rain
        hum = rng.uniform(0.7, 1.0)
        windspeed = rng.uniform(0.2, 0.6)
    else:  # heavy rain
        hum = rng.uniform(0.8, 1.0)
        windspeed = rng.uniform(0.3, 0.8)

    return round(hum, 4), round(windspeed, 4)


def calculate_bike_counts(
    hour: int, workingday: int, holiday: int, weather: int, temp: float, season: int
) -> tuple[int, int, int]:
    """Calculate casual, registered, and total bike counts"""
    # Base demand factors
    hour_factor: float = {
        0: 0.18,
        1: 0.15,
        2: 0.08,
        3: 0.05,
        4: 0.12,
        5: 0.15,
        6: 0.2,
        7: 0.5,
        8: 0.8,
        9: 0.6,
        10: 0.5,
        11: 0.6,
        12: 0.7,
        13: 0.7,
        14: 0.6,
        15: 0.6,
        16: 0.7,
        17: 0.9,
        18: 0.8,
        19: 0.6,
        20: 0.4,
        21: 0.3,
        22: 0.2,
        23: 0.15,
    }[hour]

    # Working day vs weekend patterns
    if workingday == 1:  # working day
        if hour in [7, 8, 17, 18]:  # rush hours
            registered_multiplier = 2.5
            casual_multiplier = 0.8
        else:
            registered_multiplier = 1.0
            casual_multiplier = 0.6
    else:  # weekend/holiday
        if 10 <= hour <= 16:  # leisure hours
            registered_multiplier = 0.7
            casual_multiplier = 2.0
        else:
            registered_multiplier = 0.5
            casual_multiplier = 1.0

    # Weather impact
    weather_impact = max(0.1, 1.2 - (weather - 1) * 0.3)

    # Temperature impact (people prefer moderate temperatures)
    temp_impact: float = 1.0 - abs(temp - 0.6) * 0.8
    temp_impact = max(0.2, temp_impact)

    # Season impact
    season_multiplier: float = {1: 1.1, 2: 1.3, 3: 1.2, 4: 0.8}[season]

    # Holiday impact
    holiday_impact: float = 0.7 if holiday == 1 else 1.0

    # Calculate base counts
    base_registered: float = (
        50
        * hour_factor
        * registered_multiplier
        * weather_impact
        * temp_impact
        * season_multiplier
        * holiday_impact
    )
    base_casual: float = (
        20
        * hour_factor
        * casual_multiplier
        * weather_impact
        * temp_impact
        * season_multiplier
        * holiday_impact
    )

    # Add some randomness
    registered: int = max(
        0, int(base_registered + rng.normal(0, base_registered * 0.3))
    )
    casual: int = max(0, int(base_casual + rng.normal(0, base_casual * 0.4)))

    return casual, registered, casual + registered


def is_holiday(date: datetime) -> bool:
    """Detect US federal holidays using the holidays package"""
    return date.date() in us_holidays


def generate_bike_data(start_date: str, end_date: str) -> pl.DataFrame:
    """Generate bike rental data between start and end dates"""
    start: datetime = datetime.strptime(start_date, "%Y-%m-%d")
    end: datetime = datetime.strptime(end_date, "%Y-%m-%d")

    data_rows: list[dict[str, Any]] = []
    current: datetime = start

    print(f"Generating data from {start_date} to {end_date}...")

    while current <= end:
        # Generate 24 hours of data for current date
        for hour in range(24):
            dt = current + timedelta(hours=hour)

            # Basic date/time features
            season = get_season(dt.month)
            yr = dt.year - 2011  # 0-based year (2011=0, 2012=1, 2024=13, etc.)
            mnth = dt.month
            hr = hour
            weekday = dt.weekday()  # 0=Monday, 6=Sunday
            holiday = 1 if is_holiday(dt) else 0
            workingday = 1 if (weekday < 5 and holiday == 0) else 0

            # Weather and environmental features
            weathersit = get_weather_situation()
            temp, atemp = calculate_temp_atemp(hour, mnth, weathersit)
            hum, windspeed = calculate_humidity_windspeed(weathersit)

            # Bike rental counts
            casual, registered, cnt = calculate_bike_counts(
                hr, workingday, holiday, weathersit, temp, season
            )

            data_rows.append(
                {
                    "datetime": dt.strftime("%Y-%m-%d %H:%M:%S"),
                    "season": season,
                    "yr": yr,
                    "mnth": mnth,
                    "hr": hr,
                    "holiday": holiday,
                    "weekday": weekday,
                    "workingday": workingday,
                    "weathersit": weathersit,
                    "temp": temp,
                    "atemp": atemp,
                    "hum": hum,
                    "windspeed": windspeed,
                    "casual": casual,
                    "registered": registered,
                    "cnt": cnt,
                }
            )

        current += timedelta(days=1)

    print(f"Generated {len(data_rows)} hourly records")

    # Create Polars DataFrame with proper schema
    df = pl.DataFrame(data_rows)

    # Ensure proper data types
    return df.with_columns(
        [
            pl.col("datetime").str.to_datetime("%Y-%m-%d %H:%M:%S"),
            pl.col("season").cast(pl.Int64),
            pl.col("yr").cast(pl.Int64),
            pl.col("mnth").cast(pl.Int64),
            pl.col("hr").cast(pl.Int64),
            pl.col("holiday").cast(pl.Int64),
            pl.col("weekday").cast(pl.Int64),
            pl.col("workingday").cast(pl.Int64),
            pl.col("weathersit").cast(pl.Int64),
            pl.col("temp").cast(pl.Float64),
            pl.col("atemp").cast(pl.Float64),
            pl.col("hum").cast(pl.Float64),
            pl.col("windspeed").cast(pl.Float64),
            pl.col("casual").cast(pl.Int64),
            pl.col("registered").cast(pl.Int64),
            pl.col("cnt").cast(pl.Int64),
        ]
    )


def main() -> int:
    """Main function to parse arguments and generate bike rental data."""
    parser = argparse.ArgumentParser(description="Generate realistic bike rental data")
    parser.add_argument("--start", required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", required=True, help="End date (YYYY-MM-DD)")
    parser.add_argument(
        "--output", default="bike_data.parquet", help="Output file path"
    )

    args = parser.parse_args()

    try:
        # Validate dates
        datetime.strptime(args.start, "%Y-%m-%d")
        datetime.strptime(args.end, "%Y-%m-%d")
    except ValueError:
        print("Error: Dates must be in YYYY-MM-DD format")
        return 1

    # Generate data
    df = generate_bike_data(args.start, args.end)

    # Save to file
    output_path = Path(args.output)
    if output_path.suffix.lower() == ".parquet":
        df.write_parquet(output_path)
    elif output_path.suffix.lower() == ".csv":
        df.write_csv(output_path)
    else:
        # Default to parquet
        output_path = output_path.with_suffix(".parquet")
        df.write_parquet(output_path)

    print(f"Data saved to: {output_path}")
    print(f"Shape: {df.shape}")
    print("\nFirst few rows:")
    print(df.head())

    return 0


if __name__ == "__main__":
    exit(main())
