from pathlib import Path
from typing import Any

import torch

from app.core.config import get_settings
from app.core.logging import get_logger
from app.ml.datasets.transform import MNIST_MEAN, MNIST_STD
from app.ml.models.mnist_model import MNISTModel

logger = get_logger(__name__)


def get_prediction(input_tensor: torch.Tensor) -> int:
    """Run a fresh, untrained ``MNISTModel`` on one input tensor.

    Parameters
    ----------
    input_tensor : torch.Tensor
        Shape ``(1, 1, 28, 28)``.

    Returns
    -------
    int
        Predicted digit (0-9).
    """
    model = MNISTModel()
    model.eval()
    with torch.no_grad():
        output = model(input_tensor)
        return int(output.argmax(dim=1, keepdim=True))


def load_checkpoint(
    checkpoint_path: Path | None,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Load a saved model state dict, defaulting to the latest checkpoint.

    Parameters
    ----------
    checkpoint_path : Path, optional
        Defaults to ``checkpoint_dir/model_latest.pth``.

    Returns
    -------
    tuple of (dict, dict)
        ``(state_dict, metadata)``, where ``metadata`` holds
        ``checkpoint_path`` plus ``normalize_mean``/``normalize_std``.

    Notes
    -----
    Checkpoints written by ``app/ml/train/train.py`` are a dict with
    ``state_dict`` plus the exact normalization used at training time, so a
    later change to ``MNIST_MEAN``/``MNIST_STD`` can't silently mismatch an
    existing checkpoint at serving time. A checkpoint saved as a bare state
    dict instead (pre-dates this format, or was written directly rather
    than via ``train.py``) falls back to today's ``MNIST_MEAN``/
    ``MNIST_STD`` with a logged warning rather than failing.
    """
    settings = get_settings()
    path = checkpoint_path or settings.checkpoint_dir / "model_latest.pth"
    # weights_only=True restricts unpickling to tensors/state dicts/basic
    # Python types — a checkpoint file is a plausible supply-chain attack
    # vector otherwise.
    loaded = torch.load(path, map_location="cpu", weights_only=True)

    if isinstance(loaded, dict) and "state_dict" in loaded:
        state_dict = loaded["state_dict"]
        normalize_mean = loaded.get("normalize_mean", MNIST_MEAN)
        normalize_std = loaded.get("normalize_std", MNIST_STD)
    else:
        state_dict = loaded
        normalize_mean, normalize_std = MNIST_MEAN, MNIST_STD
        logger.warning(
            f"{path} has no transform metadata (predates this checkpoint "
            "format) — assuming today's default MNIST normalization."
        )

    return state_dict, {
        "checkpoint_path": str(path),
        "normalize_mean": normalize_mean,
        "normalize_std": normalize_std,
    }
