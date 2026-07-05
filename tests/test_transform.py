from app.ml.datasets.transform import (
    MNIST_MEAN,
    MNIST_STD,
    dataset_fingerprint,
    get_mnist_transform,
)


def test_get_mnist_transform_uses_canonical_constants() -> None:
    transform = get_mnist_transform()

    normalize = transform.transforms[-1]
    assert normalize.mean == MNIST_MEAN
    assert normalize.std == MNIST_STD


def test_dataset_fingerprint_is_stable() -> None:
    assert dataset_fingerprint() == dataset_fingerprint()


def test_dataset_fingerprint_is_short_hex() -> None:
    fingerprint = dataset_fingerprint()

    assert len(fingerprint) == 12
    int(fingerprint, 16)  # raises ValueError if not valid hex
