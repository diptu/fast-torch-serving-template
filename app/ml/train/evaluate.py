import argparse
import sys
from pathlib import Path

from app.core.config import get_settings
from app.core.logging import get_logger, setup_logging
from app.ml.datasets.loader import get_data_loaders
from app.ml.inference.predict import load_checkpoint
from app.ml.models.mnist_model import MNISTModel
from app.ml.train.engine import ClassificationReport, evaluate_detailed

settings = get_settings()
logger = get_logger(__name__)


def evaluate_checkpoint(
    checkpoint_path: Path | None, batch_size: int, device: str
) -> ClassificationReport:
    """Load a checkpoint and evaluate it against the validation set.

    Parameters
    ----------
    checkpoint_path : Path, optional
        Defaults to ``checkpoint_dir/model_latest.pth``.
    batch_size : int
    device : str
        ``"cuda"``, ``"mps"``, or ``"cpu"``.

    Returns
    -------
    ClassificationReport
        val_loss/val_accuracy plus confusion matrix, per-class precision/
        recall/F1, and expected calibration error — this is the canonical
        pre-promotion report, not just the two scalars `make train` tracks
        per epoch.

    Raises
    ------
    SystemExit
        If the checkpoint path doesn't exist or fails to load.
    """

    if checkpoint_path and not checkpoint_path.exists():
        logger.error(f"Checkpoint not found at: {checkpoint_path}")
        sys.exit(1)

    logger.info(f"Loading checkpoint from: {checkpoint_path or 'latest'}")

    try:
        model = MNISTModel().to(device)
        state_dict, _metadata = load_checkpoint(checkpoint_path)
        model.load_state_dict(state_dict)
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        sys.exit(1)

    _, val_loader = get_data_loaders(batch_size=batch_size)

    logger.info("Running evaluation...")
    report = evaluate_detailed(
        model, val_loader, device, num_classes=settings.num_classes
    )

    logger.info(
        "Evaluation complete",
        extra={
            "extra_data": {
                "val_loss": report.val_loss,
                "val_accuracy": report.val_accuracy,
                "expected_calibration_error": report.expected_calibration_error,
                "per_class": report.per_class,
                "confusion_matrix": report.confusion_matrix,
            }
        },
    )
    return report


def main() -> None:
    setup_logging(settings.log_level)
    parser = argparse.ArgumentParser(description="Evaluate model checkpoints.")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=settings.BATCH_SIZE)
    parser.add_argument("--device", default=settings.device)
    args = parser.parse_args()

    report = evaluate_checkpoint(args.checkpoint, args.batch_size, args.device)

    # Print metrics to stdout for external observability tools (like Makefiles)
    print(
        f"VAL_LOSS={report.val_loss:.4f} VAL_ACCURACY={report.val_accuracy:.4f} "
        f"ECE={report.expected_calibration_error:.4f}"
    )


if __name__ == "__main__":
    main()
