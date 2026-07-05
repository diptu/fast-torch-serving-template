import pytest
import torch

from app.ml.export_onnx import export_onnx, main
from app.ml.models.mnist_model import MNISTModel

try:
    import onnx
except ImportError:
    onnx = None

requires_onnx = pytest.mark.skipif(
    onnx is None, reason="onnx export deps are optional: uv sync --group onnx"
)


@requires_onnx
def test_export_onnx_produces_a_valid_model(tmp_path) -> None:
    checkpoint_path = tmp_path / "model_latest.pth"
    torch.save(MNISTModel().state_dict(), checkpoint_path)
    output_path = tmp_path / "model.onnx"

    result = export_onnx(checkpoint_path, output_path)

    assert result == output_path
    assert output_path.exists()

    exported = onnx.load(str(output_path))
    onnx.checker.check_model(exported)
    assert [i.name for i in exported.graph.input] == ["image"]
    assert [o.name for o in exported.graph.output] == ["log_probabilities"]


def test_main_exits_when_checkpoint_missing(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        "sys.argv",
        ["export_onnx.py", "--checkpoint", str(tmp_path / "missing.pth")],
    )

    with pytest.raises(SystemExit) as exc_info:
        main()

    assert exc_info.value.code == 1
