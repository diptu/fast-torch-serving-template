import torch

from app.core.config import get_settings
from app.ml.inference.predict import get_prediction

SETTINGS = get_settings()


def test_mnist_inference_shape() -> None:
    # Create a dummy input tensor (batch_size, channels, height, width)
    dummy_input = torch.randn(1, 1, 28, 28)
    prediction = get_prediction(dummy_input)
    assert 0 <= prediction <= SETTINGS.num_classes
