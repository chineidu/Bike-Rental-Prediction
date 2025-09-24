# Bike Rental Data Generator

This script generates realistic hourly bike rental data with the following features:

## Features Generated

- **datetime**: Hourly timestamps between start and end dates
- **season**: Season (1=spring, 2=summer, 3=fall, 4=winter)
- **yr**: Year (0-based, 2011=0)
- **mnth**: Month (1-12)
- **hr**: Hour (0-23)
- **holiday**: Holiday indicator (0/1)
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

### Generate data for one year

```bash
uv run data_gen.py --start "2011-01-01" --end "2011-12-31" --output "bike_data_2011.parquet"
```

### Generate data for a few months

```bash
uv run data_gen.py --start "2012-03-01" --end "2012-08-31" --output "spring_summer_2012.parquet"
```

### Generate CSV instead of Parquet

```bash
uv run data_gen.py --start "2011-01-01" --end "2011-01-31" --output "january_2011.csv"
```

## Data Characteristics

The generator creates realistic patterns:

- **Rush hour peaks**: Higher registered users during 7-9am and 5-7pm on working days
- **Leisure peaks**: Higher casual users during 10am-4pm on weekends
- **Weather effects**: Lower ridership during poor weather conditions
- **Temperature effects**: Optimal ridership around moderate temperatures
- **Seasonal variations**: Higher usage in spring/summer, lower in winter
- **Holiday effects**: Reduced ridership on holidays

## Sample Output

```
Shape: (8760, 16)  # One year of hourly data
┌─────────────────────┬────────┬─────┬──────┬───┬───────────┬────────┬────────────┬─────┐
│ datetime            ┆ season ┆ yr  ┆ mnth ┆ … ┆ windspeed ┆ casual ┆ registered ┆ cnt │
│ 2011-01-01 00:00:00 ┆ 4      ┆ 0   ┆ 1    ┆ … ┆ 0.1283    ┆ 3      ┆ 13         ┆ 16  │
│ 2011-01-01 01:00:00 ┆ 4      ┆ 0   ┆ 1    ┆ … ┆ 0.378     ┆ 8      ┆ 32         ┆ 40  │
└─────────────────────┴────────┴─────┴──────┴───┴───────────┴────────┴────────────┴─────┘
```
