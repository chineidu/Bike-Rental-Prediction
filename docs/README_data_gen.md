# Bike Rental Data Generator

## Table of Contents
<!-- TOC -->

- [Bike Rental Data Generator](#bike-rental-data-generator)
  - [Table of Contents](#table-of-contents)
  - [Features Generated](#features-generated)
  - [Usage Examples](#usage-examples)
    - [Generate data for any year](#generate-data-for-any-year)
    - [Generate data for specific periods](#generate-data-for-specific-periods)
    - [Generate CSV instead of Parquet](#generate-csv-instead-of-parquet)
  - [Data Characteristics](#data-characteristics)
  - [Key Features](#key-features)
    - [✅ Year Flexibility](#-year-flexibility)
    - [✅ Comprehensive Holiday Support](#-comprehensive-holiday-support)
    - [✅ Realistic Patterns](#-realistic-patterns)
  - [Sample Output](#sample-output)
  - [Dependencies](#dependencies)
  - [Schema Compatibility](#schema-compatibility)

<!-- /TOC -->
This script generates realistic hourly bike rental data for **any year** with comprehensive US federal holiday support. Compatible with the original UCI Bike Sharing Dataset schema while extending to modern dates.

## Features Generated

- **datetime**: Hourly timestamps between start and end dates
- **season**: Season (1=spring, 2=summer, 3=fall, 4=winter)
- **yr**: Year (0-based, 2011=0, 2012=1, 2024=13, etc.)
- **mnth**: Month (1-12)
- **hr**: Hour (0-23)
- **holiday**: US federal holiday indicator (0/1) - uses `holidays` package
- **weekday**: Day of week (0=Monday, 6=Sunday)
- **workingday**: Working day indicator (0/1)
- **weathersit**: Weather situation (1=clear, 2=mist, 3=light rain, 4=heavy rain)
- **temp**: Normalized temperature (0-1)
- **atemp**: Normalized feeling temperature (0-1)
- **hum**: Normalized humidity (0-1)
- **windspeed**: Normalized windspeed (0-1)
- **casual**: Count of casual users
- **registered**: Count of registered users
- **cnt**: Total count (casual + registered)

## Usage Examples

### Generate data for any year

```bash
# Original dataset years
uv run data_gen.py --start "2011-01-01" --end "2011-12-31" --output "bike_data_2011.parquet"

# Modern data (2024)
uv run data_gen.py --start "2024-01-01" --end "2024-12-31" --output "bike_data_2024.parquet"

# Multi-year dataset
uv run data_gen.py --start "2020-01-01" --end "2024-12-31" --output "bikes_2020_2024.parquet"
```

### Generate data for specific periods

```bash
# Summer 2025
uv run data_gen.py --start "2025-06-01" --end "2025-08-31" --output "summer_2025.parquet"

# Holiday season 2024
uv run data_gen.py --start "2024-11-01" --end "2024-12-31" --output "holidays_2024.parquet"
```

### Generate CSV instead of Parquet

```bash
uv run data_gen.py --start "2024-01-01" --end "2024-01-31" --output "january_2024.csv"
```

## Data Characteristics

The generator creates realistic patterns:

- **Rush hour peaks**: Higher registered users during 7-9am and 5-7pm on working days
- **Leisure peaks**: Higher casual users during 10am-4pm on weekends
- **Weather effects**: Lower ridership during poor weather conditions
- **Temperature effects**: Optimal ridership around moderate temperatures
- **Seasonal variations**: Higher usage in spring/summer, lower in winter
- **Holiday effects**: Reduced ridership on comprehensive US federal holidays

## Key Features

### ✅ Year Flexibility

- **Any year supported**: Generate data for 2011, 2024, 2030, or any other year
- **Maintains schema**: 100% compatible with original UCI dataset structure
- **Year encoding**: Continues pattern (2011=0, 2012=1, 2024=13, etc.)

### ✅ Comprehensive Holiday Support

- **US Federal Holidays**: New Year's, MLK Day, Presidents Day, Memorial Day, Independence Day, Labor Day, Columbus Day, Veterans Day, Thanksgiving, Christmas
- **Observed dates**: Handles holidays that fall on weekends (e.g., Christmas observed)
- **Dynamic updates**: Uses `holidays` Python package for accuracy across all years

### ✅ Realistic Patterns

- **Temporal consistency**: Correct weekdays, seasons, and calendar logic for any year
- **Behavioral patterns**: Same rush hour, weekend, and holiday patterns regardless of year
- **Weather simulation**: Consistent seasonal temperature and weather patterns

## Sample Output

```txt
# Original dataset format (2011)
Shape: (8760, 16)  # One year of hourly data
┌─────────────────────┬────────┬─────┬──────┬───┬───────────┬────────┬────────────┬─────┐
│ datetime            ┆ season ┆ yr  ┆ mnth ┆ … ┆ windspeed ┆ casual ┆ registered ┆ cnt │
│ 2011-01-01 00:00:00 ┆ 4      ┆ 0   ┆ 1    ┆ … ┆ 0.1283    ┆ 3      ┆ 13         ┆ 16  │
│ 2011-01-01 01:00:00 ┆ 4      ┆ 0   ┆ 1    ┆ … ┆ 0.378     ┆ 8      ┆ 32         ┆ 40  │
└─────────────────────┴────────┴─────┴──────┴───┴───────────┴────────┴────────────┴─────┘

# Modern data format (2024)
┌─────────────────────┬────────┬─────┬──────┬───┬───────────┬────────┬────────────┬─────┐
│ datetime            ┆ season ┆ yr  ┆ mnth ┆ … ┆ windspeed ┆ casual ┆ registered ┆ cnt │
│ 2024-01-01 00:00:00 ┆ 4      ┆ 13  ┆ 1    ┆ … ┆ 0.1283    ┆ 1      ┆ 0          ┆ 1   │
│ 2024-07-04 12:00:00 ┆ 2      ┆ 13  ┆ 7    ┆ … ┆ 0.2156    ┆ 15     ┆ 8          ┆ 23  │
└─────────────────────┴────────┴─────┴──────┴───┴───────────┴────────┴────────────┴─────┘
```

## Dependencies

```bash
# Core packages (already included in your environment)
uv add polars pyarrow pandas pydantic-settings rich holidays
```

## Schema Compatibility

The generated data maintains 100% compatibility with the original UCI Bike Sharing Dataset:

- Same 16-column structure
- Identical data types and value ranges
- Compatible with existing analysis code and models
- Extends seamlessly from 2011-2012 to any modern year

For detailed year compatibility information, see [YEAR_COMPATIBILITY.md](YEAR_COMPATIBILITY.md).
