import mlflow
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

import app.ml.train.train as train_module
from app.ml.datasets.transform import MNIST_MEAN, MNIST_STD


def _tiny_loader() -> DataLoader:
    x = torch.randn(4, 1, 28, 28)
    y = torch.randint(0, 10, (4,))
    return DataLoader(TensorDataset(x, y), batch_size=2)


@pytest.fixture
def fast_training_setup(monkeypatch, tmp_path):
    """Redirects training to tiny synthetic data, one epoch, and an isolated
    tmp mlflow store/checkpoint dir so the test is fast and doesn't touch
    the real project's mlruns/checkpoints."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        train_module,
        "get_data_loaders",
        lambda batch_size: (_tiny_loader(), _tiny_loader()),
    )
    monkeypatch.setattr(train_module.settings, "EPOCHS", 1)
    monkeypatch.setattr(
        train_module.settings,
        "mlflow_tracking_uri",
        f"sqlite:///{tmp_path / 'mlflow.db'}",
    )
    return tmp_path


def test_set_seed_makes_torch_rand_reproducible() -> None:
    train_module.set_seed(123)
    first = torch.rand(3)
    train_module.set_seed(123)
    second = torch.rand(3)
    assert torch.equal(first, second)


def test_train_saves_checkpoint(fast_training_setup) -> None:
    tmp_path = fast_training_setup

    train_module.train()

    checkpoint_dir = tmp_path / "checkpoints"
    assert (checkpoint_dir / "model_latest.pth").exists()

    # A run-tagged copy is also kept so a bad run can be rolled back from.
    versioned = [
        p for p in checkpoint_dir.glob("model_*.pth") if p.name != "model_latest.pth"
    ]
    assert len(versioned) == 1


def test_train_logs_params_and_metrics_to_mlflow(fast_training_setup) -> None:
    train_module.train()

    mlflow.set_tracking_uri(train_module.settings.mlflow_tracking_uri)
    experiment = mlflow.get_experiment_by_name(
        train_module.settings.mlflow_experiment_name
    )
    runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])

    assert len(runs) == 1
    assert runs.iloc[0]["params.epochs"] == "1"
    assert "metrics.val_accuracy" in runs.columns
    assert "metrics.train_loss" in runs.columns
    assert "metrics.val_expected_calibration_error" in runs.columns
    assert "tags.dataset_fingerprint" in runs.columns


def test_train_saves_checkpoint_with_transform_metadata(fast_training_setup) -> None:
    tmp_path = fast_training_setup

    train_module.train()

    checkpoint = torch.load(
        tmp_path / "checkpoints" / "model_latest.pth",
        map_location="cpu",
        weights_only=True,
    )
    assert set(checkpoint) == {"state_dict", "normalize_mean", "normalize_std"}
    assert checkpoint["normalize_mean"] == MNIST_MEAN
    assert checkpoint["normalize_std"] == MNIST_STD
