from fastapi import APIRouter

router = APIRouter()


@router.post("/predict")
async def predict() -> dict[str, str]:
    """Endpoint for making predictions."""
    return {"message": "Prediction endpoint"}
