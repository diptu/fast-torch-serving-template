import torch
import torch.nn.functional as F  # noqa: N812
from torch import nn


class MNISTModel(nn.Module):
    """Small CNN classifier for 28x28 grayscale MNIST digits (2 conv + 2 fc)."""

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Batch of images, shape ``(N, 1, 28, 28)``.

        Returns
        -------
        torch.Tensor
            Log-probabilities per class, shape ``(N, 10)``.
        """
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
