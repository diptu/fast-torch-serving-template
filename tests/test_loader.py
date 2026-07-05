import torch
from torch.utils.data import Dataset, RandomSampler, SequentialSampler

from app.ml.datasets import loader as loader_module


class _FakeMNIST(Dataset):
    """Stands in for torchvision.datasets.MNIST so tests don't hit the network."""

    def __init__(self, root, train, download, transform, **kwargs) -> None:
        self.train = train
        self.samples = [(torch.zeros(1, 28, 28), 0) for _ in range(4)]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        return self.samples[idx]


def test_get_data_loaders_wires_train_and_val_loaders(monkeypatch) -> None:
    monkeypatch.setattr(loader_module.datasets, "MNIST", _FakeMNIST)

    train_loader, val_loader = loader_module.get_data_loaders(batch_size=2)

    assert train_loader.batch_size == 2
    assert val_loader.batch_size == 2
    assert train_loader.dataset.train is True
    assert val_loader.dataset.train is False
    assert isinstance(train_loader.sampler, RandomSampler)
    assert isinstance(val_loader.sampler, SequentialSampler)
