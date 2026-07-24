import ast
import base64
import json
import signal
import tarfile
from io import BytesIO

import pytest

import scripts.train_dispatch as train_dispatch_module
from scripts.run_remote import NoRemoteKernelError
from scripts.train_dispatch import (
    CHECKPOINT_MARKER,
    MLFLOW_MARKER,
    REPO_ROOT,
    _build_remote_script,
    _collect_sources,
    _raise_keyboard_interrupt,
    _save_colab_mlflow_data,
    dispatch_local,
    dispatch_to_colab,
    main,
)


def test_collect_sources_includes_app_init_and_bundle_dirs() -> None:
    files = _collect_sources()

    assert "app/__init__.py" in files
    assert any(path.startswith("app/core/") for path in files)
    assert any(path.startswith("app/ml/") for path in files)
    assert all("__pycache__" not in path for path in files)
    assert files["app/__init__.py"] == (REPO_ROOT / "app" / "__init__.py").read_text()


def test_build_remote_script_round_trips_real_bundle() -> None:
    """Uses the real, current app/core + app/ml source (not a toy string) so
    this also catches file content that would break naive string-templating
    (quotes, backslashes, parens) — anything real Python source can contain."""
    files = _collect_sources()
    script = _build_remote_script(files)

    assert CHECKPOINT_MARKER in script
    assert "app.ml.train.train import train" in script

    tree = ast.parse(script)  # also proves the generated script is valid Python
    loads_call = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "loads"
    )
    embedded = loads_call.args[0]
    assert isinstance(embedded, ast.Constant)
    assert json.loads(embedded.value) == files


def test_raise_keyboard_interrupt_includes_signal_number() -> None:
    with pytest.raises(KeyboardInterrupt, match=str(signal.SIGTERM)):
        _raise_keyboard_interrupt(signal.SIGTERM, None)


def test_dispatch_to_colab_falls_back_when_no_kernel_published(monkeypatch) -> None:
    def _raise(topic: str, stale_after: int):
        raise NoRemoteKernelError("nothing published")

    monkeypatch.setattr("scripts.train_dispatch.fetch_latest_connection", _raise)
    previous_handler = signal.getsignal(signal.SIGTERM)

    result = dispatch_to_colab("some-topic", 20 * 60)

    assert result is False
    # Must be restored regardless of how dispatch exits, not left pointing
    # at the dispatch-time handler (see the docstring on dispatch_to_colab
    # for why this matters — a real incident during development).
    assert signal.getsignal(signal.SIGTERM) is previous_handler


def test_dispatch_local_calls_train(monkeypatch) -> None:
    calls = []
    monkeypatch.setattr("app.ml.train.train.train", lambda: calls.append(True))

    dispatch_local()

    assert calls == [True]


def test_main_no_colab_skips_dispatch_to_colab(monkeypatch) -> None:
    monkeypatch.setattr("sys.argv", ["train_dispatch.py", "--no-colab"])
    colab_calls = []
    local_calls = []
    monkeypatch.setattr(
        "scripts.train_dispatch.dispatch_to_colab",
        lambda *a, **k: colab_calls.append(True) or True,
    )
    monkeypatch.setattr(
        "scripts.train_dispatch.dispatch_local", lambda: local_calls.append(True)
    )

    main()

    assert colab_calls == []
    assert local_calls == [True]


def _make_tarball(files: dict[str, bytes]) -> bytes:
    buf = BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for name, content in files.items():
            info = tarfile.TarInfo(name=name)
            info.size = len(content)
            tar.addfile(info, BytesIO(content))
    return buf.getvalue()


def test_save_colab_mlflow_data_extracts_into_its_own_directory(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.setattr(train_dispatch_module, "REPO_ROOT", tmp_path)
    tarball = _make_tarball(
        {"mlflow.db": b"fake-sqlite-bytes", "mlruns/meta.txt": b"fake-artifact"}
    )

    dest = _save_colab_mlflow_data(tarball)

    assert dest.parent == tmp_path / "colab_runs"
    assert (dest / "mlflow.db").read_bytes() == b"fake-sqlite-bytes"
    assert (dest / "mlruns" / "meta.txt").read_bytes() == b"fake-artifact"


def test_dispatch_to_colab_success_saves_checkpoint_and_mlflow_data(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.setattr(train_dispatch_module, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(
        train_dispatch_module,
        "fetch_latest_connection",
        lambda topic, stale_after: ("http://fake-colab", "faketoken"),
    )

    class _FakeResponse:
        def json(self) -> dict[str, str]:
            return {"id": "kernel123"}

    monkeypatch.setattr(
        train_dispatch_module.requests, "post", lambda *a, **k: _FakeResponse()
    )
    monkeypatch.setattr(train_dispatch_module.requests, "delete", lambda *a, **k: None)
    # _collect_sources() has its own dedicated test against the real repo;
    # here we only care about dispatch_to_colab's orchestration.
    monkeypatch.setattr(
        train_dispatch_module, "_collect_sources", lambda: {"app/__init__.py": ""}
    )

    class _FakeWebSocket:
        def close(self) -> None:
            pass

    monkeypatch.setattr(
        train_dispatch_module.websocket,
        "create_connection",
        lambda *a, **k: _FakeWebSocket(),
    )

    checkpoint_bytes = b"fake-model-weights"
    mlflow_tarball = _make_tarball({"mlflow.db": b"fake-sqlite"})
    fake_stdout = (
        MLFLOW_MARKER
        + base64.b64encode(mlflow_tarball).decode()
        + "\n"
        + CHECKPOINT_MARKER
        + base64.b64encode(checkpoint_bytes).decode()
    )
    monkeypatch.setattr(
        train_dispatch_module,
        "run_cell",
        lambda ws, session_id, script: (True, fake_stdout),
    )

    result = dispatch_to_colab("some-topic", 20 * 60)

    assert result is True
    assert (
        tmp_path / "checkpoints" / "model_latest.pth"
    ).read_bytes() == checkpoint_bytes
    colab_dirs = list((tmp_path / "colab_runs").iterdir())
    assert len(colab_dirs) == 1
    assert (colab_dirs[0] / "mlflow.db").read_bytes() == b"fake-sqlite"


def test_main_falls_back_to_local_when_colab_dispatch_fails(monkeypatch) -> None:
    monkeypatch.setattr("sys.argv", ["train_dispatch.py"])
    monkeypatch.setattr(
        "scripts.train_dispatch.dispatch_to_colab", lambda *a, **k: False
    )
    local_calls = []
    monkeypatch.setattr(
        "scripts.train_dispatch.dispatch_local", lambda: local_calls.append(True)
    )

    main()

    assert local_calls == [True]
