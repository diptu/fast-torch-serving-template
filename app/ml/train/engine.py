import itertools
from dataclasses import dataclass

import numpy as np
import torch
from sklearn.metrics import confusion_matrix as sk_confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from torch import nn, optim
from torch.amp.autocast_mode import autocast
from torch.amp.grad_scaler import GradScaler
from torch.utils.data import DataLoader

from app.core.logging import get_logger

logger = get_logger(__name__)


def train_one_epoch(  # noqa: PLR0913 — mixed-precision training needs all of these
    model: nn.Module,
    loader: DataLoader[tuple[torch.Tensor, int]],
    optimizer: optim.Optimizer,
    device: str,
    scaler: GradScaler,
    max_grad_norm: float = 1.0,
) -> dict[str, float]:
    """Run one training epoch with mixed-precision and gradient clipping.

    Parameters
    ----------
    model : nn.Module
        Model to train, in-place.
    loader : DataLoader
        Yields ``(inputs, targets)`` batches.
    optimizer : optim.Optimizer
        Optimizer stepped once per batch.
    device : str
        ``"cuda"``, ``"mps"``, or ``"cpu"``.
    scaler : GradScaler
        Must be constructed with ``enabled=(device == "cuda")`` — reused
        below to gate autocast too, so the two can't drift out of sync.
    max_grad_norm : float, default 1.0
        Gradient-clipping threshold.

    Returns
    -------
    dict of str to float
        ``{"train_loss": ..., "train_accuracy": ...}``.

    Notes
    -----
    Mixed precision only does anything on CUDA — autocast has no fp16/bf16
    path for CPU/MPS here.
    """
    model.train()
    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    autocast_device = "cuda" if device == "cuda" else "cpu"

    for x, y in loader:
        inputs, targets = x.to(device), y.to(device)

        optimizer.zero_grad(set_to_none=True)

        # Mixed precision context (no-op unless scaler/device are CUDA)
        with autocast(device_type=autocast_device, enabled=scaler.is_enabled()):
            logits = model(inputs)
            loss = criterion(logits, targets)

        scaler.scale(loss).backward()

        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * inputs.size(0)
        total_correct += (logits.argmax(dim=1) == targets).sum().item()
        total_samples += inputs.size(0)

    return {
        "train_loss": total_loss / total_samples,
        "train_accuracy": total_correct / total_samples,
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader[tuple[torch.Tensor, int]],
    device: str,
) -> dict[str, float]:
    """Evaluate loss/accuracy on a validation set, no gradient tracking.

    Parameters
    ----------
    model : nn.Module
        Model to evaluate.
    loader : DataLoader
        Yields ``(inputs, targets)`` batches.
    device : str
        ``"cuda"``, ``"mps"``, or ``"cpu"``.

    Returns
    -------
    dict of str to float
        ``{"val_loss": ..., "val_accuracy": ...}``.
    """
    model.eval()
    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for x, y in loader:
        inputs, targets = x.to(device), y.to(device)
        logits = model(inputs)
        loss = criterion(logits, targets)

        total_loss += loss.item() * inputs.size(0)
        total_correct += (logits.argmax(dim=1) == targets).sum().item()
        total_samples += inputs.size(0)

    return {
        "val_loss": total_loss / total_samples,
        "val_accuracy": total_correct / total_samples,
    }


@dataclass
class ClassificationReport:
    """Everything `evaluate()` above doesn't tell you: aggregate accuracy
    hides exactly the failure modes this reports on directly — a class
    whose recall collapsed even though overall accuracy held, or a model
    that's accurate but badly overconfident on the cases it gets wrong."""

    val_loss: float
    val_accuracy: float
    confusion_matrix: list[list[int]]
    per_class: dict[int, dict[str, float]]
    expected_calibration_error: float


@torch.no_grad()
def evaluate_detailed(
    model: nn.Module,
    loader: DataLoader[tuple[torch.Tensor, int]],
    device: str,
    num_classes: int = 10,
    num_bins: int = 10,
) -> ClassificationReport:
    """Confusion matrix, per-class precision/recall/F1, and calibration error.

    Parameters
    ----------
    model : nn.Module
    loader : DataLoader
        Yields ``(inputs, targets)`` batches.
    device : str
        ``"cuda"``, ``"mps"``, or ``"cpu"``.
    num_classes : int, default 10
        Forces a full ``num_classes`` x ``num_classes`` confusion matrix
        (and one precision/recall/F1 entry per class) even if a class
        happens not to appear in ``loader`` — otherwise sklearn would infer
        a smaller shape from whichever labels show up, making results
        incomparable across runs/batches.
    num_bins : int, default 10
        Confidence buckets for the calibration error calculation.

    Returns
    -------
    ClassificationReport

    Notes
    -----
    A separate, heavier pass from ``evaluate()`` above — that one runs every
    epoch and only needs loss/accuracy; this is meant to be read by a human
    or a promotion gate once, on the final model (see train.py), not
    tracked epoch over epoch.
    """
    model.eval()
    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    total_samples = 0
    all_preds: list[int] = []
    all_targets: list[int] = []
    all_confidences: list[float] = []
    all_correct: list[bool] = []

    for x, y in loader:
        inputs, targets = x.to(device), y.to(device)
        logits = model(inputs)
        loss = criterion(logits, targets)
        total_loss += loss.item() * inputs.size(0)
        total_samples += inputs.size(0)

        probs = torch.softmax(logits, dim=1)
        confidences, preds = probs.max(dim=1)
        all_preds.extend(preds.cpu().tolist())
        all_targets.extend(targets.cpu().tolist())
        all_confidences.extend(confidences.cpu().tolist())
        all_correct.extend((preds == targets).cpu().tolist())

    labels = list(range(num_classes))
    confusion = sk_confusion_matrix(all_targets, all_preds, labels=labels).tolist()
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_targets, all_preds, labels=labels, zero_division=0
    )
    per_class = {
        cls: {
            "precision": float(precision[cls]),
            "recall": float(recall[cls]),
            "f1": float(f1[cls]),
        }
        for cls in labels
    }

    return ClassificationReport(
        val_loss=total_loss / total_samples,
        val_accuracy=sum(all_correct) / total_samples,
        confusion_matrix=confusion,
        per_class=per_class,
        expected_calibration_error=_expected_calibration_error(
            all_confidences, all_correct, num_bins
        ),
    )


def _expected_calibration_error(
    confidences: list[float], correct: list[bool], num_bins: int
) -> float:
    """Bucket predictions by confidence, weigh each bucket's
    |confidence - accuracy| gap by its share of samples, and sum.

    Parameters
    ----------
    confidences : list of float
    correct : list of bool
    num_bins : int

    Returns
    -------
    float
        In ``[0, 1]`` — 0 means confidence always matched actual accuracy.
    """
    confidences_arr = np.asarray(confidences)
    correct_arr = np.asarray(correct, dtype=float)
    total = len(confidences_arr)
    bin_edges = np.linspace(0.0, 1.0, num_bins + 1)

    error = 0.0
    for lo, hi in itertools.pairwise(bin_edges):
        in_bin = (confidences_arr > lo) & (confidences_arr <= hi)
        bin_count = int(in_bin.sum())
        if bin_count == 0:
            continue
        bin_accuracy = float(correct_arr[in_bin].mean())
        bin_confidence = float(confidences_arr[in_bin].mean())
        error += (bin_count / total) * abs(bin_accuracy - bin_confidence)
    return error
