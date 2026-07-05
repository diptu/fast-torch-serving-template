import io
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from PIL import Image, UnidentifiedImageError

from app.ml.models.mnist_model import MNISTModel
from app.services.inference_service import (
    PREDICTION_CLASS_TOTAL,
    PREDICTION_CONFIDENCE,
    PREDICTION_SHADOW_AGREEMENT_TOTAL,
    InferenceService,
    get_inference_service,
)


def _biased_model(target_digit: int) -> MNISTModel:
    """A model whose prediction is target_digit regardless of input — zeroing
    fc2's weight means only its bias affects the output, so a single very
    large bias entry deterministically wins argmax every time."""
    model = MNISTModel()
    torch.nn.init.zeros_(model.fc2.weight)
    torch.nn.init.zeros_(model.fc2.bias)
    with torch.no_grad():
        model.fc2.bias[target_digit] = 10.0
    model.eval()
    return model


def _sample_value(metric, name: str, labels: dict[str, str] | None = None) -> float:
    """0.0 if the sample doesn't exist yet — a labeled Counter/Histogram
    child (e.g. a specific `agreement` label) isn't created until first
    used, unlike the metric itself."""
    for family in metric.collect():
        for sample in family.samples:
            if sample.name == name and (labels is None or sample.labels == labels):
                return sample.value
    return 0.0


def _png_bytes() -> bytes:
    image = Image.new("L", (28, 28), color=200)
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


def _patch_checkpoint_dir(monkeypatch, checkpoint_dir: Path) -> None:
    fake_settings = SimpleNamespace(checkpoint_dir=checkpoint_dir)
    monkeypatch.setattr(
        "app.services.inference_service.get_settings", lambda: fake_settings
    )


def test_init_without_checkpoint_uses_untrained_model(monkeypatch, tmp_path) -> None:
    _patch_checkpoint_dir(monkeypatch, tmp_path)

    service = InferenceService()

    assert service.model.training is False


def test_init_with_checkpoint_loads_saved_weights(monkeypatch, tmp_path) -> None:
    trained = MNISTModel()
    torch.nn.init.constant_(trained.fc2.bias, 3.14)
    torch.save(trained.state_dict(), tmp_path / "model_latest.pth")
    _patch_checkpoint_dir(monkeypatch, tmp_path)

    service = InferenceService()

    assert torch.allclose(service.model.fc2.bias, trained.fc2.bias)


def test_init_with_checkpoint_uses_its_own_saved_transform(
    monkeypatch, tmp_path
) -> None:
    torch.save(
        {
            "state_dict": MNISTModel().state_dict(),
            "normalize_mean": (0.5,),
            "normalize_std": (0.25,),
        },
        tmp_path / "model_latest.pth",
    )
    _patch_checkpoint_dir(monkeypatch, tmp_path)

    service = InferenceService()

    normalize = service.transform.transforms[-1]
    assert normalize.mean == (0.5,)
    assert normalize.std == (0.25,)


def test_predict_image_returns_valid_probability_distribution(
    monkeypatch, tmp_path
) -> None:
    _patch_checkpoint_dir(monkeypatch, tmp_path)
    service = InferenceService()

    predicted_digit, probabilities = service.predict_image(_png_bytes())

    assert 0 <= predicted_digit <= 9
    assert len(probabilities) == 10
    assert probabilities[predicted_digit] == max(probabilities)
    assert sum(probabilities) == pytest.approx(1.0, abs=1e-4)


def test_predict_image_raises_for_invalid_bytes(monkeypatch, tmp_path) -> None:
    _patch_checkpoint_dir(monkeypatch, tmp_path)
    service = InferenceService()

    with pytest.raises(UnidentifiedImageError):
        service.predict_image(b"not an image")


def test_get_inference_service_is_a_cached_singleton() -> None:
    assert get_inference_service() is get_inference_service()


def test_init_without_checkpoint_sets_checkpoint_loaded_false(
    monkeypatch, tmp_path
) -> None:
    _patch_checkpoint_dir(monkeypatch, tmp_path)

    service = InferenceService()

    assert service.checkpoint_loaded is False


def test_predict_image_records_confidence_and_class_metrics(
    monkeypatch, tmp_path
) -> None:
    _patch_checkpoint_dir(monkeypatch, tmp_path)
    service = InferenceService()
    count_before = _sample_value(PREDICTION_CONFIDENCE, "predict_confidence_count")

    predicted_digit, _ = service.predict_image(_png_bytes())

    assert (
        _sample_value(PREDICTION_CONFIDENCE, "predict_confidence_count")
        == count_before + 1
    )
    class_count = _sample_value(
        PREDICTION_CLASS_TOTAL,
        "predict_class_total",
        labels={"digit": str(predicted_digit)},
    )
    assert class_count >= 1


def test_predict_batch_records_metrics_per_image(monkeypatch, tmp_path) -> None:
    _patch_checkpoint_dir(monkeypatch, tmp_path)
    service = InferenceService()
    count_before = _sample_value(PREDICTION_CONFIDENCE, "predict_confidence_count")

    results = service.predict_batch([_png_bytes(), _png_bytes()])

    assert (
        _sample_value(PREDICTION_CONFIDENCE, "predict_confidence_count")
        == count_before + 2
    )
    for predicted_digit, _ in results:
        assert (
            _sample_value(
                PREDICTION_CLASS_TOTAL,
                "predict_class_total",
                labels={"digit": str(predicted_digit)},
            )
            >= 1
        )


def test_reload_picks_up_a_newly_written_checkpoint(monkeypatch, tmp_path) -> None:
    _patch_checkpoint_dir(monkeypatch, tmp_path)
    service = InferenceService()
    assert service.checkpoint_loaded is False

    trained = MNISTModel()
    torch.nn.init.constant_(trained.fc2.bias, 9.99)
    torch.save(trained.state_dict(), tmp_path / "model_latest.pth")

    reloaded = service.reload()

    assert reloaded is True
    assert service.checkpoint_loaded is True
    assert torch.allclose(service.model.fc2.bias, trained.fc2.bias)


def test_init_without_shadow_file_sets_shadow_loaded_false(
    monkeypatch, tmp_path
) -> None:
    _patch_checkpoint_dir(monkeypatch, tmp_path)

    service = InferenceService()

    assert service.shadow_loaded is False
    assert service.shadow_model is None


def test_reload_picks_up_a_staged_shadow_checkpoint(monkeypatch, tmp_path) -> None:
    _patch_checkpoint_dir(monkeypatch, tmp_path)
    service = InferenceService()
    assert service.shadow_loaded is False

    torch.save(_biased_model(7).state_dict(), tmp_path / "model_shadow.pth")
    reloaded = service.reload()

    assert reloaded is False  # no champion checkpoint, unrelated to the shadow
    assert service.shadow_loaded is True
    assert service.shadow_model is not None


def test_reload_disables_shadow_when_file_removed(monkeypatch, tmp_path) -> None:
    _patch_checkpoint_dir(monkeypatch, tmp_path)
    shadow_path = tmp_path / "model_shadow.pth"
    torch.save(_biased_model(7).state_dict(), shadow_path)
    service = InferenceService()
    assert service.shadow_loaded is True

    shadow_path.unlink()
    service.reload()

    assert service.shadow_loaded is False
    assert service.shadow_model is None


def test_reload_handles_corrupt_shadow_checkpoint_gracefully(
    monkeypatch, tmp_path, caplog
) -> None:
    _patch_checkpoint_dir(monkeypatch, tmp_path)
    (tmp_path / "model_shadow.pth").write_bytes(b"not a checkpoint")

    service = InferenceService()

    assert service.shadow_loaded is False
    assert service.shadow_model is None
    assert "Failed to load shadow checkpoint" in caplog.text
    # The main model is unaffected by a broken shadow.
    assert service.model.training is False


def test_predict_image_records_shadow_agreement_when_matching(
    monkeypatch, tmp_path
) -> None:
    torch.save(_biased_model(3).state_dict(), tmp_path / "model_latest.pth")
    torch.save(_biased_model(3).state_dict(), tmp_path / "model_shadow.pth")
    _patch_checkpoint_dir(monkeypatch, tmp_path)
    service = InferenceService()
    before = _sample_value(
        PREDICTION_SHADOW_AGREEMENT_TOTAL,
        "predict_shadow_agreement_total",
        labels={"agreement": "match"},
    )

    predicted_digit, _ = service.predict_image(_png_bytes())

    assert predicted_digit == 3
    after = _sample_value(
        PREDICTION_SHADOW_AGREEMENT_TOTAL,
        "predict_shadow_agreement_total",
        labels={"agreement": "match"},
    )
    assert after == before + 1


def test_predict_image_records_shadow_agreement_when_mismatching(
    monkeypatch, tmp_path
) -> None:
    torch.save(_biased_model(3).state_dict(), tmp_path / "model_latest.pth")
    torch.save(_biased_model(7).state_dict(), tmp_path / "model_shadow.pth")
    _patch_checkpoint_dir(monkeypatch, tmp_path)
    service = InferenceService()
    before = _sample_value(
        PREDICTION_SHADOW_AGREEMENT_TOTAL,
        "predict_shadow_agreement_total",
        labels={"agreement": "mismatch"},
    )

    predicted_digit, _ = service.predict_image(_png_bytes())

    assert predicted_digit == 3
    after = _sample_value(
        PREDICTION_SHADOW_AGREEMENT_TOTAL,
        "predict_shadow_agreement_total",
        labels={"agreement": "mismatch"},
    )
    assert after == before + 1


def test_predict_image_shadow_scoring_failure_does_not_break_prediction(
    monkeypatch, tmp_path, caplog
) -> None:
    torch.save(_biased_model(3).state_dict(), tmp_path / "model_latest.pth")
    torch.save(_biased_model(7).state_dict(), tmp_path / "model_shadow.pth")
    _patch_checkpoint_dir(monkeypatch, tmp_path)
    service = InferenceService()

    def _boom(*args, **kwargs):
        raise RuntimeError("shadow exploded")

    monkeypatch.setattr(service.shadow_model, "forward", _boom)

    predicted_digit, probabilities = service.predict_image(_png_bytes())

    assert predicted_digit == 3
    assert len(probabilities) == 10
    assert "Shadow model scoring failed" in caplog.text


def test_predict_batch_scores_shadow_per_image(monkeypatch, tmp_path) -> None:
    torch.save(_biased_model(3).state_dict(), tmp_path / "model_latest.pth")
    torch.save(_biased_model(3).state_dict(), tmp_path / "model_shadow.pth")
    _patch_checkpoint_dir(monkeypatch, tmp_path)
    service = InferenceService()
    before = _sample_value(
        PREDICTION_SHADOW_AGREEMENT_TOTAL,
        "predict_shadow_agreement_total",
        labels={"agreement": "match"},
    )

    results = service.predict_batch([_png_bytes(), _png_bytes()])

    assert all(digit == 3 for digit, _ in results)
    after = _sample_value(
        PREDICTION_SHADOW_AGREEMENT_TOTAL,
        "predict_shadow_agreement_total",
        labels={"agreement": "match"},
    )
    assert after == before + 2
