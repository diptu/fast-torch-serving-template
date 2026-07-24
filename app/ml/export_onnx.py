"""Export a trained checkpoint to ONNX for runtimes other than PyTorch
(ONNX Runtime, TensorRT via its ONNX parser, etc).

Requires the `onnx` dependency group: `uv sync --group onnx`.

Usage:
    uv run --group onnx python -m app.ml.export_onnx
    uv run --group onnx python -m app.ml.export_onnx \\
        --checkpoint checkpoints/model_<run_id>.pth --output model.onnx
"""

import argparse
import sys
from pathlib import Path

import torch

from app.core.config import get_settings
from app.core.logging import get_logger, setup_logging
from app.ml.inference.predict import load_checkpoint
from app.ml.models.mnist_model import MNISTModel

settings = get_settings()
logger = get_logger(__name__)


def export_onnx(checkpoint_path: Path, output_path: Path) -> Path:
    """Export a checkpoint to ONNX with a dynamic batch dimension.

    Parameters
    ----------
    checkpoint_path : Path
    output_path : Path
        Where to write the ``.onnx`` file (parent dirs created if needed).

    Returns
    -------
    Path
        ``output_path``, unchanged.
    """
    model = MNISTModel()
    state_dict, _ = load_checkpoint(checkpoint_path)
    model.load_state_dict(state_dict)
    model.eval()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sample_input = torch.randn(1, 1, 28, 28)
    batch_dim = torch.export.Dim("batch_size")
    torch.onnx.export(
        model,
        (sample_input,),
        str(output_path),
        input_names=["image"],
        output_names=["log_probabilities"],
        # Key must match MNISTModel.forward's parameter name ("x"), not the
        # input_names given to the exported graph above.
        dynamic_shapes={"x": {0: batch_dim}},
    )
    logger.info(f"Exported {checkpoint_path} -> {output_path}")
    return output_path


def main() -> None:
    setup_logging(settings.log_level)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Defaults to <checkpoint_dir>/model_latest.pth",
    )
    parser.add_argument("--output", type=Path, default=Path("checkpoints/model.onnx"))
    args = parser.parse_args()

    checkpoint_path = args.checkpoint or (settings.checkpoint_dir / "model_latest.pth")
    if not checkpoint_path.exists():
        logger.error(f"Checkpoint not found at {checkpoint_path}")
        sys.exit(1)

    export_onnx(checkpoint_path, args.output)


if __name__ == "__main__":
    main()
