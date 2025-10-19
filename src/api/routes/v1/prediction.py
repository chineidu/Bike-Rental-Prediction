from datetime import datetime
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status

from src import create_logger
from src.api.utilities.ml_utils import PredictionService
from src.api.utilities.utilities import ModelManagerDict, get_model_manager
from src.config import app_config
from src.schemas import PredictedPriceResponse, RawInputSchema
from src.schemas.types import PredictionResultDict
from src.utilities.utils import async_log_function_duration

router = APIRouter(tags=["prediction"])
logger = create_logger(name="prediction")


@router.post("/predict", status_code=status.HTTP_200_OK)
@async_log_function_duration
async def predict_price_endpoint(
    input: RawInputSchema, model_manager: ModelManagerDict = Depends(get_model_manager)
) -> PredictedPriceResponse:
    """Endpoint for predicting the dynamic price."""
    try:
        service: PredictionService = model_manager["prediction_service"]
        data: list[dict[str, Any]] = input.model_dump()["data"]
        currency: str = input.model_dump()["currency"]

        base_price: float = app_config.business_config.base_price
        base_elasticity: float = app_config.business_config.base_elasticity

        logger.info("Estimating price...")

        pred: PredictionResultDict = service.predict(
            data=data,
            base_price=base_price,
            base_elasticity=base_elasticity,
            currency=currency,
        )
        capacity_components: dict[str, Any] = {
            "max_capacity": app_config.business_config.max_capacity,
            "current_utilization": pred["demand"],
            "available_bikes": app_config.business_config.max_capacity - pred["demand"],
            "utilization_rate": round(
                pred["demand"] / app_config.business_config.max_capacity, 2
            ),
        }
        price_components: dict[str, Any] = {
            "price_multiplier": pred["price_multiplier"],
            "surge": pred["surge"],
            "discount": pred["discount"],
            "base_price": pred["base_price"],
            "competitor_price": pred["competitor_price"],
            "min_price": pred["min_price"],
            "max_price": pred["max_price"],
            "currency": pred["currency"],
        }
        result: PredictedPriceResponse = PredictedPriceResponse(
            status="success",
            capacity_components=capacity_components,  # type: ignore
            price_components=price_components,  # type: ignore
            timestamp=datetime.now().isoformat(timespec="seconds"),
        )

        return result

    except Exception as e:
        logger.error(f"Prediction error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "status": "failed",
                "error": str(e),
                "timestamp": datetime.now().isoformat(timespec="seconds"),
            },
        ) from e
