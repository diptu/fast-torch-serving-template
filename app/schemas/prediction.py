from pydantic import BaseModel, Field


class PredictionResponse(BaseModel):
    """One digit classification: predicted class, its confidence, and the
    full per-class probability distribution."""

    predicted_digit: int = Field(..., ge=0, le=9)
    confidence: float = Field(..., ge=0.0, le=1.0)
    probabilities: list[float]


class BatchPredictionResponse(BaseModel):
    """One ``PredictionResponse`` per image in a batch request, same order."""

    predictions: list[PredictionResponse]
