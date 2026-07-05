import torch


def get_device() -> torch.device:
    """Pick the best available local compute device.

    Returns
    -------
    torch.device
        ``cuda`` if available, else ``mps`` (Apple Silicon), else ``cpu``.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
