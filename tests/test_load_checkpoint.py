import torch

from app.ml.datasets.transform import MNIST_MEAN, MNIST_STD
from app.ml.inference.predict import load_checkpoint
from app.ml.models.mnist_model import MNISTModel


def test_load_checkpoint_new_format_returns_saved_transform(tmp_path) -> None:
    path = tmp_path / "model_latest.pth"
    torch.save(
        {
            "state_dict": MNISTModel().state_dict(),
            "normalize_mean": (0.5,),
            "normalize_std": (0.2,),
        },
        path,
    )

    _, metadata = load_checkpoint(path)

    assert metadata["normalize_mean"] == (0.5,)
    assert metadata["normalize_std"] == (0.2,)
    assert metadata["checkpoint_path"] == str(path)


def test_load_checkpoint_bare_state_dict_falls_back_to_defaults(
    tmp_path, caplog
) -> None:
    path = tmp_path / "model_latest.pth"
    torch.save(MNISTModel().state_dict(), path)

    state_dict, metadata = load_checkpoint(path)

    assert metadata["normalize_mean"] == MNIST_MEAN
    assert metadata["normalize_std"] == MNIST_STD
    assert set(state_dict) == set(MNISTModel().state_dict())
    assert "no transform metadata" in caplog.text
