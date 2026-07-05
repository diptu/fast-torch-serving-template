import io
from functools import lru_cache
from pathlib import Path

import torch
from PIL import Image
from prometheus_client import Counter, Histogram
from torchvision import transforms

from app.core.config import get_settings
from app.core.logging import get_logger
from app.ml.datasets.transform import MNIST_MEAN, MNIST_STD
from app.ml.inference.predict import load_checkpoint
from app.ml.models.mnist_model import MNISTModel

logger = get_logger(__name__)


def _build_transform(
    mean: tuple[float, ...], std: tuple[float, ...]
) -> transforms.Compose:
    """Build the serving-time preprocessing pipeline for a given checkpoint.

    Parameters
    ----------
    mean : tuple of float
    std : tuple of float

    Returns
    -------
    transforms.Compose

    Notes
    -----
    Grayscale + Resize aren't part of training's transform (see
    ``app/ml/datasets/transform.py``) since the MNIST dataset is already
    28x28 grayscale — they're here because serving accepts arbitrary
    uploaded images that aren't guaranteed to be either.
    """
    return transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )


# Model-quality signal, distinct from the HTTP-layer metrics
# prometheus-fastapi-instrumentator already exposes on GET /metrics (see
# app/main.py) — a drifting or degraded model looks identical to a healthy
# one at the HTTP layer, but shows up here as falling confidence or a
# vanished class.
PREDICTION_CONFIDENCE = Histogram(
    "predict_confidence",
    "Predicted-class confidence (softmax probability) per prediction.",
    buckets=(0.1, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0),
)
PREDICTION_CLASS_TOTAL = Counter(
    "predict_class_total",
    "Count of predictions per predicted digit class.",
    ["digit"],
)
# Populated only while a shadow model is staged (see app/ml/train/promote.py
# stage_shadow/commit_shadow) — lets an operator watch a candidate's live
# agreement rate with the champion before committing it, without the shadow
# ever being able to affect what a client actually gets served.
PREDICTION_SHADOW_AGREEMENT_TOTAL = Counter(
    "predict_shadow_agreement_total",
    "Agreement between the champion and staged shadow model's predicted digit.",
    ["agreement"],
)


def _record_prediction_metrics(
    predicted_digit: int, probabilities: list[float]
) -> None:
    PREDICTION_CONFIDENCE.observe(probabilities[predicted_digit])
    PREDICTION_CLASS_TOTAL.labels(digit=str(predicted_digit)).inc()


def _decode_and_transform(
    image_bytes: bytes, transform: transforms.Compose
) -> torch.Tensor:
    """Decode raw image bytes and apply the MNIST preprocessing transform.

    Parameters
    ----------
    image_bytes : bytes
    transform : transforms.Compose
        The loaded checkpoint's transform (see ``InferenceService.reload``),
        not necessarily today's ``MNIST_MEAN``/``MNIST_STD`` default.

    Returns
    -------
    torch.Tensor
        Shape ``(1, 28, 28)``, normalized.
    """
    image = Image.open(io.BytesIO(image_bytes))
    tensor: torch.Tensor = transform(image)
    return tensor


def _top_prediction(probabilities: list[float]) -> int:
    """Index of the highest-probability class.

    Parameters
    ----------
    probabilities : list of float

    Returns
    -------
    int
    """
    return max(range(len(probabilities)), key=probabilities.__getitem__)


class InferenceService:
    """Loads ``MNISTModel`` weights and serves predictions on CPU.

    Access via the cached ``get_inference_service()``, not directly.
    """

    def __init__(self) -> None:
        settings = get_settings()
        self.device = torch.device("cpu")
        self.checkpoint_path: Path = settings.checkpoint_dir / "model_latest.pth"
        self.shadow_checkpoint_path: Path = settings.checkpoint_dir / "model_shadow.pth"
        self.model = MNISTModel().to(self.device)
        self.shadow_model: MNISTModel | None = None
        self.checkpoint_loaded = False
        self.shadow_loaded = False
        self.transform = _build_transform(MNIST_MEAN, MNIST_STD)
        self.shadow_transform = _build_transform(MNIST_MEAN, MNIST_STD)
        self.reload()

    def reload(self) -> bool:
        """(Re)load weights (and their transform) from ``checkpoint_path``.

        Returns
        -------
        bool
            Whether a checkpoint was actually found — an untrained model is
            still usable, just not meaningfully accurate. Safe to call again
            later to pick up a newly trained checkpoint without restarting
            the process.

        Notes
        -----
        Rebuilds ``self.transform`` from the checkpoint's own
        ``normalize_mean``/``normalize_std`` (see ``load_checkpoint``)
        rather than assuming today's ``MNIST_MEAN``/``MNIST_STD`` — a
        checkpoint trained under different normalization keeps working
        correctly even if this module's defaults later change. Also
        (re)loads the shadow model, if one is staged — see ``_reload_shadow``.
        """
        if self.checkpoint_path.exists():
            state_dict, metadata = load_checkpoint(self.checkpoint_path)
            self.model.load_state_dict(state_dict)
            self.transform = _build_transform(
                metadata["normalize_mean"], metadata["normalize_std"]
            )
            self.checkpoint_loaded = True
            logger.info(f"Loaded model weights from {self.checkpoint_path}")
        else:
            self.checkpoint_loaded = False
            logger.warning(
                f"No checkpoint found at {self.checkpoint_path}; "
                "serving predictions from an untrained model."
            )
        self.model.eval()

        self._reload_shadow()

        return self.checkpoint_loaded

    def _reload_shadow(self) -> None:
        """(Re)load the shadow model from ``shadow_checkpoint_path``, if staged.

        Notes
        -----
        Independent of the main checkpoint and always best-effort: a
        missing or corrupt shadow file disables shadow scoring (logged, not
        raised) rather than affecting serving at all — the shadow model
        never influences what ``predict_image``/``predict_batch`` return,
        so a problem with it is never a reason to fail a real request.
        """
        if not self.shadow_checkpoint_path.exists():
            self.shadow_model = None
            self.shadow_loaded = False
            return

        try:
            state_dict, metadata = load_checkpoint(self.shadow_checkpoint_path)
            shadow_model = MNISTModel().to(self.device)
            shadow_model.load_state_dict(state_dict)
            shadow_model.eval()
        except Exception:
            logger.exception(
                f"Failed to load shadow checkpoint from {self.shadow_checkpoint_path}; "
                "shadow scoring disabled."
            )
            self.shadow_model = None
            self.shadow_loaded = False
            return

        self.shadow_model = shadow_model
        self.shadow_transform = _build_transform(
            metadata["normalize_mean"], metadata["normalize_std"]
        )
        self.shadow_loaded = True
        logger.info(f"Loaded shadow model weights from {self.shadow_checkpoint_path}")

    def _score_shadow(self, image_bytes: bytes, champion_digit: int) -> None:
        """Run the staged shadow model on the same input and record agreement.

        Parameters
        ----------
        image_bytes : bytes
        champion_digit : int
            What ``model_latest.pth`` predicted, to compare the shadow against.

        Notes
        -----
        Best-effort and silent on failure by design: a broken shadow model
        must never break or alter a real prediction, since the entire point
        of shadow scoring is validating a candidate against live traffic
        without it being able to influence what's actually served.
        """
        if self.shadow_model is None:
            return

        try:
            tensor = (
                _decode_and_transform(image_bytes, self.shadow_transform)
                .unsqueeze(0)
                .to(self.device)
            )
            with torch.no_grad():
                shadow_log_probs = self.shadow_model(tensor)
                shadow_probabilities: list[float] = (
                    torch.exp(shadow_log_probs).squeeze(0).tolist()
                )
            shadow_digit = _top_prediction(shadow_probabilities)
        except Exception:
            logger.exception("Shadow model scoring failed; skipping this prediction.")
            return

        agreement = "match" if shadow_digit == champion_digit else "mismatch"
        PREDICTION_SHADOW_AGREEMENT_TOTAL.labels(agreement=agreement).inc()

    def predict_image(self, image_bytes: bytes) -> tuple[int, list[float]]:
        """Classify a single image.

        Parameters
        ----------
        image_bytes : bytes

        Returns
        -------
        tuple of (int, list of float)
            ``(predicted_digit, probabilities)``.
        """
        tensor = (
            _decode_and_transform(image_bytes, self.transform)
            .unsqueeze(0)
            .to(self.device)
        )

        with torch.no_grad():
            log_probs = self.model(tensor)
            probabilities: list[float] = torch.exp(log_probs).squeeze(0).tolist()

        predicted_digit = _top_prediction(probabilities)
        _record_prediction_metrics(predicted_digit, probabilities)
        self._score_shadow(image_bytes, predicted_digit)
        return predicted_digit, probabilities

    def predict_batch(self, images: list[bytes]) -> list[tuple[int, list[float]]]:
        """Classify a batch of images in a single forward pass.

        Parameters
        ----------
        images : list of bytes

        Returns
        -------
        list of (int, list of float)
            One ``(predicted_digit, probabilities)`` pair per image.

        Notes
        -----
        One forward pass for the whole batch instead of one per image is
        the point of a batch endpoint: it amortizes compute across images
        rather than just accepting them together and looping.
        """
        batch = torch.stack(
            [_decode_and_transform(b, self.transform) for b in images]
        ).to(self.device)

        with torch.no_grad():
            log_probs = self.model(batch)
            probabilities_batch: list[list[float]] = torch.exp(log_probs).tolist()

        results = [(_top_prediction(p), p) for p in probabilities_batch]
        for image_bytes, (predicted_digit, probabilities) in zip(
            images, results, strict=True
        ):
            _record_prediction_metrics(predicted_digit, probabilities)
            self._score_shadow(image_bytes, predicted_digit)
        return results


@lru_cache
def get_inference_service() -> InferenceService:
    """Get the process-wide cached ``InferenceService`` singleton.

    Returns
    -------
    InferenceService
    """
    return InferenceService()
