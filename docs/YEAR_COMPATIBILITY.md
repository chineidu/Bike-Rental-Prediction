# Year Compatibility Guide for Bike Data Generator

## Table of Contents
<!-- TOC -->

- [Year Compatibility Guide for Bike Data Generator](#year-compatibility-guide-for-bike-data-generator)
  - [Table of Contents](#table-of-contents)
  - [Original Dataset Context](#original-dataset-context)
  - [Extended Year Support](#extended-year-support)
    - [Year Encoding Examples](#year-encoding-examples)
    - [Usage Examples](#usage-examples)
  - [What Changes by Year](#what-changes-by-year)
  - [Model Training Considerations](#model-training-considerations)
  - [Schema Compatibility](#schema-compatibility)

<!-- /TOC -->
## Original Dataset Context

The original bike sharing dataset from UCI covers **2011-2012** only, where:

- `yr = 0` represents 2011
- `yr = 1` represents 2012

## Extended Year Support

Our data generator can create realistic bike rental data for **any year**, maintaining the same schema and encoding:

### Year Encoding Examples

```txt
2011 -> yr = 0   (original)
2012 -> yr = 1   (original)
2013 -> yr = 2
2014 -> yr = 3
...
2024 -> yr = 13
2025 -> yr = 14
2030 -> yr = 19
```

### Usage Examples

**Generate data for 2024:**

```bash
uv run data_gen.py --start "2024-01-01" --end "2024-12-31" --output "bikes_2024.parquet"
```

**Generate data for any future year:**

```bash
uv run data_gen.py --start "2030-06-01" --end "2030-08-31" --output "bikes_summer_2030.parquet"
```

**Generate historical data (pre-2011):**

```bash
uv run data_gen.py --start "2010-01-01" --end "2010-12-31" --output "bikes_2010.parquet"
# Results in yr = -1 for 2010
```

## What Changes by Year

✅ **Automatically Updated:**

- **Holidays**: US federal holidays for any year (New Year's, July 4th, Christmas, etc.)
- **Weekdays**: Correct day-of-week calculations for any date
- **Seasons**: Month-based seasons work for any year

✅ **Remains Consistent:**

- **Weather patterns**: Same realistic temperature, humidity, wind patterns
- **Usage patterns**: Same rush hour, weekend, holiday behavioral patterns
- **Data schema**: All 16 columns maintain identical structure and types

## Model Training Considerations

**For time series models**: The `yr` column captures year-over-year trends
**For general ML models**: You may want to exclude or transform the `yr` column if training on multiple years
**For validation**: Use different years for train/test splits to test temporal generalization

## Schema Compatibility

Generated data for any year maintains 100% compatibility with the original UCI dataset schema, so existing analysis code will work without modification.
