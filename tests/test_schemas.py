import pytest
from pydantic import ValidationError

from app.schemas.prediction import PredictionResponse


def test_prediction_response_accepts_valid_data() -> None:
    response = PredictionResponse(
        predicted_digit=7,
        confidence=0.98,
        probabilities=[0.01] * 9 + [0.91],
    )
    assert response.predicted_digit == 7
    assert response.confidence == 0.98
    assert len(response.probabilities) == 10


@pytest.mark.parametrize("digit", [-1, 10])
def test_prediction_response_rejects_out_of_range_digit(digit: int) -> None:
    with pytest.raises(ValidationError):
        PredictionResponse(predicted_digit=digit, confidence=0.5, probabilities=[0.5])


@pytest.mark.parametrize("confidence", [-0.1, 1.1])
def test_prediction_response_rejects_out_of_range_confidence(confidence: float) -> None:
    with pytest.raises(ValidationError):
        PredictionResponse(
            predicted_digit=0, confidence=confidence, probabilities=[1.0]
        )
