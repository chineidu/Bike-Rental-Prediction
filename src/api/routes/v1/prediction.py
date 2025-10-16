from datetime import datetime
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status

from src import create_logger
from src.api.utilities.ml_utils import (
    estimate_price,
    get_competitor_price,
    predict_user_demand,
)
from src.api.utilities.utilities import ModelManager, get_model_manager
from src.config import app_config
from src.schemas import PredictedPriceResponse, RawInputSchema
from src.utilities.utils import async_log_function_duration

router = APIRouter(tags=["prediction"])
logger = create_logger(name="prediction")


@router.post("/predict", status_code=status.HTTP_200_OK)
@async_log_function_duration
async def predict_demand_endpoint(
    input: RawInputSchema,
    model_manager: ModelManager = Depends(get_model_manager),  # noqa: B008
) -> dict[str, Any]:
    """Endpoint for predicting hourly demand."""
    try:
        data: list[dict[str, Any]] = input.model_dump()["data"]
        model: Any = model_manager.get_model()

        logger.info("Estimating user demand...")
        prediction: int = predict_user_demand(model, data)

        return {
            "status": "success",
            "prediction": prediction,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        }

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "status": "failed",
                "error": str(e),
                "timestamp": datetime.now().isoformat(timespec="seconds"),
            },
        ) from e


@router.post("/predict-price", status_code=status.HTTP_200_OK)
@async_log_function_duration
async def predict_price_endpoint(
    input: RawInputSchema,
    model_manager: ModelManager = Depends(get_model_manager),  # noqa: B008
) -> PredictedPriceResponse:
    """Endpoint for predicting the dynamic price."""
    try:
        data: list[dict[str, Any]] = input.model_dump()["data"]
        model: Any = model_manager.get_model()

        base_price: float = app_config.business_config.base_price
        base_elasticity: float = app_config.business_config.base_elasticity

        logger.info("Getting competitor price...")
        competitor_price: float = get_competitor_price(features=data[0])

        logger.info("Estimating user demand...")
        demand: int = predict_user_demand(model, data)

        logger.info("Estimating price...")
        capacity_components: dict[str, Any] = {
            "max_capacity": app_config.business_config.max_capacity,
            "current_utilization": demand,
            "available_bikes": app_config.business_config.max_capacity - demand,
            "utilization_rate": round(
                demand / app_config.business_config.max_capacity, 2
            ),
        }
        pred: PredictedPriceResponse = estimate_price(
            features=data,
            base_price=base_price,
            base_elasticity=base_elasticity,
            demand=demand,
            competitor_price=competitor_price,
        )
        pred.status = "success"
        pred.capacity_components = capacity_components  # type: ignore
        pred.price_components.competitor_price = competitor_price
        pred.timestamp = datetime.now().isoformat(timespec="seconds")

        return pred

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "status": "failed",
                "error": str(e),
                "timestamp": datetime.now().isoformat(timespec="seconds"),
            },
        ) from e
