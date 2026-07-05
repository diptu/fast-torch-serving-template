import torch
from torch import optim
from torch.amp.grad_scaler import GradScaler
from torch.utils.data import DataLoader, TensorDataset

from app.ml.models.mnist_model import MNISTModel
from app.ml.train.engine import (
    ClassificationReport,
    _expected_calibration_error,
    evaluate,
    evaluate_detailed,
    train_one_epoch,
)


def _tiny_loader() -> DataLoader:
    x = torch.randn(4, 1, 28, 28)
    y = torch.randint(0, 10, (4,))
    return DataLoader(TensorDataset(x, y), batch_size=2)


def test_train_one_epoch_returns_train_loss_and_accuracy() -> None:
    model = MNISTModel()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scaler = GradScaler(device="cpu", enabled=False)

    metrics = train_one_epoch(model, _tiny_loader(), optimizer, "cpu", scaler)

    assert set(metrics) == {"train_loss", "train_accuracy"}
    assert metrics["train_loss"] >= 0
    assert 0.0 <= metrics["train_accuracy"] <= 1.0


def test_evaluate_returns_val_loss_and_accuracy() -> None:
    model = MNISTModel()

    metrics = evaluate(model, _tiny_loader(), "cpu")

    assert set(metrics) == {"val_loss", "val_accuracy"}
    assert metrics["val_loss"] >= 0
    assert 0.0 <= metrics["val_accuracy"] <= 1.0


def test_evaluate_puts_model_in_eval_mode() -> None:
    model = MNISTModel()
    model.train()

    evaluate(model, _tiny_loader(), "cpu")

    assert model.training is False


def test_evaluate_detailed_returns_full_report() -> None:
    model = MNISTModel()

    report = evaluate_detailed(model, _tiny_loader(), "cpu", num_classes=10)

    assert isinstance(report, ClassificationReport)
    assert report.val_loss >= 0
    assert 0.0 <= report.val_accuracy <= 1.0
    assert len(report.confusion_matrix) == 10
    assert all(len(row) == 10 for row in report.confusion_matrix)
    assert set(report.per_class) == set(range(10))
    assert all(
        {"precision", "recall", "f1"} == set(metrics)
        for metrics in report.per_class.values()
    )
    assert 0.0 <= report.expected_calibration_error <= 1.0


def test_evaluate_detailed_confusion_matrix_diagonal_on_perfect_predictions() -> None:
    class _PerfectModel(torch.nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # Cheat: encode the true label as the batch index (see loader
            # below) and return a one-hot logit vector for it, so accuracy
            # is deterministically 100% and the confusion matrix is
            # verifiably diagonal-only.
            batch_size = x.shape[0]
            labels = torch.arange(batch_size) % 10
            return torch.nn.functional.one_hot(labels, num_classes=10).float() * 10

    x = torch.randn(10, 1, 28, 28)
    y = torch.arange(10)
    loader = DataLoader(TensorDataset(x, y), batch_size=10)

    report = evaluate_detailed(_PerfectModel(), loader, "cpu", num_classes=10)

    assert report.val_accuracy == 1.0
    for i, row in enumerate(report.confusion_matrix):
        assert row[i] == 1
        assert sum(row) == 1
    # softmax of a one-hot*10 logit is near-1.0 confidence, not exactly —
    # so ECE is near-zero rather than exactly zero.
    assert report.expected_calibration_error < 0.001


def test_expected_calibration_error_zero_when_confidence_matches_accuracy() -> None:
    # All predictions in one bucket at 0.5 confidence, exactly half correct.
    confidences = [0.5] * 10
    correct = [True] * 5 + [False] * 5

    error = _expected_calibration_error(confidences, correct, num_bins=10)

    assert error == 0.0


def test_expected_calibration_error_positive_when_overconfident() -> None:
    # Always 99% confident, but only ever right half the time.
    confidences = [0.99] * 10
    correct = [True] * 5 + [False] * 5

    error = _expected_calibration_error(confidences, correct, num_bins=10)

    assert error > 0.4
