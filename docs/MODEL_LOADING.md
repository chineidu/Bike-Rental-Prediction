# Model Loading Guide

This guide explains how to load trained models that have been saved by the ML pipeline.

## Overview

The ML pipeline saves models using the `download_model` function in `src/exp_tracking/model_loader.py`. Models are saved in different formats depending on their library:

- **scikit-learn models**: `.pkl` files (saved with `joblib`, compress=3)
- **XGBoost models**: `.json` files (native format)
- **LightGBM models**: `.txt` files (native format)

## Loading Models

### Using the Utility Function

The easiest way to load a model is using the `load_model_from_disk` function from `src.utilities.utils`:

```python
from src.utilities.utils import load_model_from_disk

# Load any model type (automatically detects format)
model = load_model_from_disk("/opt/airflow/artifacts/models/ModelType.RANDOM_FOREST_staging_1.pkl")

# Make predictions
predictions = model.predict(X_test)
```

### Manual Loading

#### scikit-learn Models (.pkl)

```python
import joblib

# Load sklearn model
model = joblib.load("/opt/airflow/artifacts/models/ModelType.RANDOM_FOREST_staging_1.pkl")
predictions = model.predict(X_test)
```

#### XGBoost Models (.json)

```python
import xgboost as xgb

# Load XGBoost model
model = xgb.Booster()
model.load_model("/opt/airflow/artifacts/models/ModelType.XGBOOST_staging_1.json")

# Make predictions (requires DMatrix)
dmatrix = xgb.DMatrix(X_test)
predictions = model.predict(dmatrix)
```

#### LightGBM Models (.txt)

```python
import lightgbm as lgb

# Load LightGBM model
model = lgb.Booster(model_file="/opt/airflow/artifacts/models/ModelType.LIGHTGBM_staging_1.txt")
predictions = model.predict(X_test)
```

## Why joblib?

We switched from `cloudpickle` to `joblib` for scikit-learn models because:

1. ✅ **Industry Standard**: `joblib` is the recommended serialization library for scikit-learn
2. ✅ **Better Performance**: Optimized for large numpy arrays (common in ML models)
3. ✅ **Compression**: Built-in compression (`compress=3`) reduces file sizes
4. ✅ **Production Ready**: More widely used and tested in production environments
5. ✅ **Compatibility**: Better cross-platform and cross-version compatibility

## Model File Locations

Models are saved in: `/opt/airflow/artifacts/models/`

File naming convention: `{model_name}_{version}.{extension}`

Example: `ModelType.RANDOM_FOREST_staging_1.pkl`

## Example: Loading in an API

```python
from pathlib import Path
from src.utilities.utils import load_model_from_disk

# Define model path
MODELS_DIR = Path("/opt/airflow/artifacts/models")
MODEL_FILE = "ModelType.RANDOM_FOREST_staging_1.pkl"
MODEL_PATH = MODELS_DIR / MODEL_FILE

# Load model once at startup
model = load_model_from_disk(MODEL_PATH)

# Use in API endpoint
@app.post("/predict")
def predict(features: dict):
    X = prepare_features(features)
    predictions = model.predict(X)
    return {"prediction": predictions.tolist()}
```

## Troubleshooting

### FileNotFoundError

**Problem**: Model file not found
**Solution**: Check that the DAG's `model_download_task` has completed successfully

### Version Mismatch

**Problem**: Model was saved with a different version of scikit-learn/xgboost/lightgbm
**Solution**: Ensure your environment has compatible library versions

### Import Error

**Problem**: Missing library (joblib, xgboost, lightgbm)
**Solution**: Install required library:

```bash
pip install joblib xgboost lightgbm
```

## Best Practices

1. **Load Once**: Load models once at application startup, not on every request
2. **Error Handling**: Wrap model loading in try-except blocks
3. **Version Tracking**: Keep track of which model version is deployed
4. **Monitoring**: Log model loading success/failure for debugging
5. **Caching**: Consider caching predictions for frequently requested inputs

## Related Files

- Model saving logic: `src/exp_tracking/model_loader.py` (`download_model` function)
- Model loading utility: `src/utilities/utils.py` (`load_model_from_disk` function)
- DAG download task: `airflow/dags/ml_dag.py` (`model_download_task`)
