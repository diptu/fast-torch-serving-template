from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

import app.ml.train.evaluate as evaluate_module
from app.ml.models.mnist_model import MNISTModel
from app.ml.train.engine import ClassificationReport

_DUMMY_REPORT = ClassificationReport(
    val_loss=0.1234,
    val_accuracy=0.9876,
    confusion_matrix=[[0] * 10 for _ in range(10)],
    per_class={i: {"precision": 0.0, "recall": 0.0, "f1": 0.0} for i in range(10)},
    expected_calibration_error=0.05,
)


def _tiny_loader() -> DataLoader:
    x = torch.randn(4, 1, 28, 28)
    y = torch.randint(0, 10, (4,))
    return DataLoader(TensorDataset(x, y), batch_size=2)


def test_evaluate_checkpoint_exits_when_path_missing(tmp_path) -> None:
    missing = tmp_path / "does_not_exist.pth"

    with pytest.raises(SystemExit) as exc_info:
        evaluate_module.evaluate_checkpoint(missing, batch_size=2, device="cpu")

    assert exc_info.value.code == 1


def test_evaluate_checkpoint_exits_when_load_fails(monkeypatch) -> None:
    def _boom(checkpoint_path):
        raise RuntimeError("corrupt checkpoint")

    monkeypatch.setattr(evaluate_module, "load_checkpoint", _boom)

    with pytest.raises(SystemExit) as exc_info:
        evaluate_module.evaluate_checkpoint(None, batch_size=2, device="cpu")

    assert exc_info.value.code == 1


def test_evaluate_checkpoint_happy_path(monkeypatch) -> None:
    state_dict = MNISTModel().state_dict()
    monkeypatch.setattr(
        evaluate_module, "load_checkpoint", lambda path: (state_dict, {})
    )
    monkeypatch.setattr(
        evaluate_module, "get_data_loaders", lambda batch_size: (None, _tiny_loader())
    )

    report = evaluate_module.evaluate_checkpoint(None, batch_size=2, device="cpu")

    assert isinstance(report, ClassificationReport)
    assert 0.0 <= report.val_accuracy <= 1.0


def test_main_prints_metrics(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        evaluate_module,
        "evaluate_checkpoint",
        lambda checkpoint_path, batch_size, device: _DUMMY_REPORT,
    )
    monkeypatch.setattr(
        "sys.argv", ["evaluate.py", "--checkpoint", "checkpoints/model_latest.pth"]
    )

    evaluate_module.main()

    captured = capsys.readouterr()
    assert "VAL_LOSS=0.1234" in captured.out
    assert "VAL_ACCURACY=0.9876" in captured.out
    assert "ECE=0.0500" in captured.out


def test_main_default_checkpoint_arg_is_none(monkeypatch) -> None:
    seen: dict[str, Path | None] = {}

    def _fake_evaluate_checkpoint(checkpoint_path, batch_size, device):
        seen["checkpoint_path"] = checkpoint_path
        return _DUMMY_REPORT

    monkeypatch.setattr(
        evaluate_module, "evaluate_checkpoint", _fake_evaluate_checkpoint
    )
    monkeypatch.setattr("sys.argv", ["evaluate.py"])

    evaluate_module.main()

    assert seen["checkpoint_path"] is None
