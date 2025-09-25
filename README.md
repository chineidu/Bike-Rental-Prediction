# Bike-Rental-Prediction

This project optimizes a city-wide bike-sharing platform's dynamic pricing. It uses a machine learning model to forecast hourly bike rental demand and a real-time algorithm to adjust prices. The goal is to maximize revenue during peak times and stimulate demand during slow periods.

## Table of Contents
<!-- TOC -->

- [Bike-Rental-Prediction](#bike-rental-prediction)
  - [Table of Contents](#table-of-contents)
  - [Installation](#installation)
    - [Install UV](#install-uv)
    - [Virtual Environment Management](#virtual-environment-management)
  - [Problem Statement](#problem-statement)
  - [Business Objectives](#business-objectives)
  - [Key Deliverables](#key-deliverables)
  - [Data Generator](#data-generator)

<!-- /TOC -->
## Installation

### Install UV

- Install UV for dependency management [UV Installation Guide](https://docs.astral.sh/uv/getting-started/installation/#__tabbed_1_1).

### Virtual Environment Management

```sh
uv venv -p "3.12"

# Install dependencies
uv sync
```

## Problem Statement

A bike-sharing platform operating across an entire city wants to optimize its dynamic pricing system.
To do so, it must accurately forecast the total number of bike rentals expected in the next hour city-wide.

These forecasts will drive real-time pricing adjustments to:

- Maximize revenue during surges
- Stimulate demand during slow periods

## Business Objectives

Develop a real-time machine learning model that forecasts hourly bike rental demand across the city, enabling:

- Dynamic pricing adjustments to maximize revenue during peak times
- Demand stimulation during off-peak periods
- Consolidate real-time data streams to collect and process bike rental data efficiently in real-time.

## Key Deliverables

- Develop and deploy a machine learning model to forecast hourly bike rental demand across the city.
- Implement a real-time dynamic pricing algorithm that adjusts prices based on forecasted demand.
- Use experiment tracking to log model performance.
- Collect and process real-time data streams to ensure the model has access to the latest rental data.
- Create a user-friendly dashboard to visualize demand forecasts and pricing adjustments.
- Ensure the system can handle high-frequency data updates and provide timely predictions.
- Model performance monitoring to track prediction accuracy and system performance over time.
- Evaluate and optimize pricing impact using A/B testing to assess the effectiveness of dynamic pricing strategies.
- Build pipelines for data ingestion, model training, and deployment to ensure seamless integration and continuous improvement of the forecasting system.

## Data Generator

This project includes a data generator script (`data_gen.py`) that creates realistic bike rental data for any year, extending beyond the original 2011-2012 dataset. The generator supports comprehensive US federal holiday recognition and maintains compatibility with the original dataset schema.

For detailed year compatibility information, see [YEAR_COMPATIBILITY.md](docs/YEAR_COMPATIBILITY.md).

For usage examples, see [README_data_gen.md](docs/README_data_gen.md).
