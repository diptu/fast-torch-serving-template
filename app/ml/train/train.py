import random

import mlflow
import mlflow.pytorch
import numpy as np
import torch
from torch import optim
from torch.amp.grad_scaler import GradScaler

from app.core.config import get_settings
from app.core.logging import get_logger, setup_logging
from app.ml.datasets.loader import get_data_loaders
from app.ml.datasets.transform import MNIST_MEAN, MNIST_STD, dataset_fingerprint
from app.ml.models.mnist_model import MNISTModel
from app.ml.train.engine import evaluate, evaluate_detailed, train_one_epoch
from app.ml.utils.device import get_device

settings = get_settings()
logger = get_logger(__name__)


def set_seed(seed: int = 42) -> None:
    """Seed ``random``/``numpy``/``torch`` (incl. CUDA) for reproducibility.

    Parameters
    ----------
    seed : int, default 42
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def train() -> None:
    """Train ``MNISTModel`` for ``settings.EPOCHS`` and save the checkpoint.

    Reads all hyperparameters from ``get_settings()``; logs params/metrics
    to MLflow and writes ``checkpoint_dir/model_latest.pth`` (plus a
    run-tagged copy) at the end.
    """
    setup_logging(settings.log_level)
    set_seed(settings.SEED)
    device = get_device()
    device_str = str(device)
    logger.info(f"Starting training on device: {device_str}")

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.mlflow_experiment_name)

    train_loader, val_loader = get_data_loaders(batch_size=settings.BATCH_SIZE)

    model = MNISTModel().to(device)
    optimizer = optim.Adam(model.parameters(), lr=settings.LEARNING_RATE)
    # Mixed precision only helps on CUDA (see engine.train_one_epoch) — an
    # enabled-but-inert scaler on CPU/MPS would just add per-step overhead
    # for no benefit, so it's disabled there.
    scaler = GradScaler(
        device="cuda" if device_str == "cuda" else "cpu", enabled=device_str == "cuda"
    )

    with mlflow.start_run() as run:
        mlflow.log_params(
            {
                "batch_size": settings.BATCH_SIZE,
                "learning_rate": settings.LEARNING_RATE,
                "epochs": settings.EPOCHS,
                "seed": settings.SEED,
                "device": device_str,
            }
        )
        # Not a real data hash (MNIST is a fixed public dataset, nothing to
        # pin) — the pattern this establishes is what matters: a queryable
        # record of what dataset + preprocessing this run used, ready for
        # when that's no longer a hardcoded, unchanging download.
        mlflow.set_tag("dataset_fingerprint", dataset_fingerprint())

        for epoch in range(settings.EPOCHS):
            train_metrics = train_one_epoch(
                model, train_loader, optimizer, device_str, scaler
            )
            val_metrics = evaluate(model, val_loader, device_str)

            logger.info(
                f"Epoch {epoch + 1}/{settings.EPOCHS} | "
                f"train_loss={train_metrics['train_loss']:.4f} "
                f"train_accuracy={train_metrics['train_accuracy']:.4f} | "
                f"val_loss={val_metrics['val_loss']:.4f} "
                f"val_accuracy={val_metrics['val_accuracy']:.4f}"
            )
            mlflow.log_metrics(
                {
                    "train_loss": train_metrics["train_loss"],
                    "train_accuracy": train_metrics["train_accuracy"],
                    "val_loss": val_metrics["val_loss"],
                    "val_accuracy": val_metrics["val_accuracy"],
                },
                step=epoch,
            )

        # Once, on the final model — not tracked per-epoch like the loop
        # above, since it's meant to be read by a human or the promotion
        # gate (app/ml/train/promote.py), not watched trend over epochs.
        report = evaluate_detailed(
            model, val_loader, device_str, num_classes=settings.num_classes
        )
        mlflow.log_dict(
            {"confusion_matrix": report.confusion_matrix}, "confusion_matrix.json"
        )
        mlflow.log_dict(
            {str(cls): metrics for cls, metrics in report.per_class.items()},
            "per_class_metrics.json",
        )
        mlflow.log_metric(
            "val_expected_calibration_error", report.expected_calibration_error
        )
        # Recall specifically (not precision/f1) is also logged as its own
        # scalar metric per class, not just inside the per_class_metrics.json
        # artifact above — app/ml/train/promote.py's per-class gate needs it
        # queryable via MlflowClient.get_run(...).data.metrics, which only
        # sees logged metrics, not artifact contents.
        mlflow.log_metrics(
            {
                f"val_recall_class_{cls}": metrics["recall"]
                for cls, metrics in report.per_class.items()
            }
        )

        # 4. Persistence — model_latest.pth is what the API serves by
        # default; the run-tagged copy preserves history so a bad training
        # run can be rolled back to by pointing APP_CHECKPOINT_DIR/a manual
        # copy at a specific model_<run_id>.pth instead of model_latest.pth.
        checkpoint_dir = settings.checkpoint_dir
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        latest_path = checkpoint_dir / "model_latest.pth"
        versioned_path = checkpoint_dir / f"model_{run.info.run_id}.pth"
        # Bundling the transform alongside the weights means a later change
        # to MNIST_MEAN/MNIST_STD can't silently mismatch this checkpoint at
        # serving time — see app/ml/inference/predict.py::load_checkpoint.
        checkpoint = {
            "state_dict": model.state_dict(),
            "normalize_mean": MNIST_MEAN,
            "normalize_std": MNIST_STD,
        }
        torch.save(checkpoint, latest_path)
        torch.save(checkpoint, versioned_path)
        logger.info(f"Model saved to {latest_path} (versioned: {versioned_path.name})")

        mlflow.log_artifact(str(latest_path))

        # "pickle" avoids mlflow's default 'pt2' (torch.export) serialization,
        # which traces the model graph and is brittle for dynamic batch sizes.
        # registered_model_name creates a new MLflow Model Registry version
        # for this run under settings.mlflow_registered_model_name — see
        # app/ml/train/promote.py, which points a "champion" alias at one.
        sample_input = torch.randn(1, 1, 28, 28)
        mlflow.pytorch.log_model(
            model.to("cpu"),
            name="model",
            input_example=sample_input.numpy(),
            serialization_format="pickle",
            registered_model_name=settings.mlflow_registered_model_name,
        )


if __name__ == "__main__":
    train()
