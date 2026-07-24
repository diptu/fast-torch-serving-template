import os

import torch
from torch.utils.data import DataLoader
from torchvision import datasets

from app.core.config import get_settings
from app.ml.datasets.transform import get_mnist_transform

settings = get_settings()

_MNISTSample = tuple[torch.Tensor, int]

# Matches the previous hardcoded default on typical dev machines, while
# avoiding over-subscribing machines/CI runners with fewer cores.
_DEFAULT_NUM_WORKERS = min(4, os.cpu_count() or 1)


def get_data_loaders(
    batch_size: int | None = None,
    num_workers: int | None = None,
) -> tuple[DataLoader[_MNISTSample], DataLoader[_MNISTSample]]:
    """Build the MNIST train/validation ``DataLoader`` pair.

    Parameters
    ----------
    batch_size : int, optional
        Defaults to ``settings.BATCH_SIZE``, read at call time.
    num_workers : int, optional
        Defaults to ``min(4, os.cpu_count())``.

    Returns
    -------
    tuple of DataLoader
        ``(train_loader, val_loader)``.

    Notes
    -----
    `batch_size`/`num_workers` default to `None` rather than binding
    `settings.BATCH_SIZE` directly as the parameter default — a default
    value is evaluated once, at function-definition time, so binding it
    directly would freeze whatever `settings.BATCH_SIZE` was when this
    module was first imported instead of tracking the current setting.
    """
    resolved_batch_size = settings.BATCH_SIZE if batch_size is None else batch_size
    resolved_num_workers = _DEFAULT_NUM_WORKERS if num_workers is None else num_workers
    # Only useful when transferring batches to a CUDA device — wasted (and
    # torch warns about it) on CPU/MPS-only runs.
    pin_memory = torch.cuda.is_available()

    transform = get_mnist_transform()

    train_dataset = datasets.MNIST(
        root=settings.data_dir, train=True, download=True, transform=transform
    )
    val_dataset = datasets.MNIST(
        root=settings.data_dir, train=False, download=True, transform=transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=resolved_batch_size,
        shuffle=True,
        num_workers=resolved_num_workers,
        pin_memory=pin_memory,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=resolved_batch_size,
        shuffle=False,
        num_workers=resolved_num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader
