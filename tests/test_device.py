import torch

from app.ml.utils import device as device_module


def test_get_device_prefers_cuda(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert device_module.get_device() == torch.device("cuda")


def test_get_device_falls_back_to_mps(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    assert device_module.get_device() == torch.device("mps")


def test_get_device_falls_back_to_cpu(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    assert device_module.get_device() == torch.device("cpu")
