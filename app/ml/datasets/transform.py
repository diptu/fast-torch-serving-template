"""Canonical MNIST preprocessing constants.

Shared by training (`app/ml/datasets/loader.py`) and serving
(`app/services/inference_service.py`) so the two can't independently drift
apart — hardcoding the same normalization twice is exactly how a training
change silently mismatches serving. `app/ml/train/train.py` also embeds
these values in every saved checkpoint, so a checkpoint's expected
normalization travels with its weights instead of being inferred from
whatever this module happens to define at serving time (see
`app/ml/inference/predict.py::load_checkpoint`).
"""

import hashlib

from torchvision import transforms

MNIST_MEAN = (0.1307,)
MNIST_STD = (0.3081,)


def get_mnist_transform() -> transforms.Compose:
    """The transform training's ``DataLoader`` applies to raw MNIST images.

    Returns
    -------
    transforms.Compose
    """
    return transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(MNIST_MEAN, MNIST_STD)]
    )


def dataset_fingerprint() -> str:
    """A short, stable identifier for "what data + preprocessing produced this run".

    Returns
    -------
    str
        First 12 hex chars of a sha256 over the dataset identity and
        normalization constants.

    Notes
    -----
    Not a full data hash — MNIST is a fixed public dataset downloaded live
    (see ``app/ml/datasets/loader.py``), so there's nothing to pin today.
    The value is in the pattern: logged as an MLflow tag per run
    (``app/ml/train/train.py``), it establishes where a real hash (of
    actual source file checksums) would plug in once this template points
    at a dataset that can actually change between runs.
    """
    payload = f"MNIST|mean={MNIST_MEAN}|std={MNIST_STD}"
    return hashlib.sha256(payload.encode()).hexdigest()[:12]
