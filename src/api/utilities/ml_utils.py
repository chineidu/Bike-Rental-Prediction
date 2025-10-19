from typing import Any

from src.api.utilities.model_loader.loader import Data, ModelInterface
from src.ml.dynamic_pricing import (
    calculate_final_price,
    get_competitor_price,
    predict_user_demand,
)
from src.schemas.types import PredictionResultDict


class PredictionService:
    def __init__(self, model: ModelInterface) -> None:
        self._model = model

    def predict(
        self, data: Data, base_price: float, base_elasticity: float, currency: str
    ) -> PredictionResultDict:
        """Get prediction from the injected model."""
        demand = self._predict_user_demand(data)
        competitor_price = self._get_competitor_price(data)
        return self._estimate_price(
            data=data,
            base_price=base_price,
            base_elasticity=base_elasticity,
            demand=demand,
            competitor_price=competitor_price,
            currency=currency,
        )

    def batch_predict(self, data_list: list[Data]) -> list[Any]:
        """Get batch predictions from the injected model."""
        return [self._model.predict(data) for data in data_list]

    def _predict_user_demand(self, data: Data) -> int:
        """Make a demand prediction using the provided model and features."""
        return predict_user_demand(self._model, data)

    def _estimate_price(
        self,
        data: Data,
        base_price: float,
        base_elasticity: float,
        demand: float,
        competitor_price: float,
        currency: str,
    ) -> PredictionResultDict:
        """Make a price prediction using the provided model and data."""
        return calculate_final_price(
            data=data,
            base_price=base_price,
            base_elasticity=base_elasticity,
            demand=demand,
            competitor_price=competitor_price,
            currency=currency,
        )

    def _get_competitor_price(self, data: Data) -> float:
        """Get competitor price based on data."""
        return get_competitor_price(data)
