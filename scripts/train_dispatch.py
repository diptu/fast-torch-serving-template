"""
train_dispatch.py
==================
Entry point for `make train`. Runs `app.ml.train.train.train()` on the best
compute backend available, in priority order:

  1. Colab GPU — if a live kernel is published on the ntfy.sh bridge (see
     colab_server.py / run_remote.py), bundle the `app.core` / `app.ml`
     source as text, execute training on that remote GPU kernel, and pull
     the resulting checkpoint back down to ./checkpoints/model_latest.pth.
  2. Local GPU — CUDA or Apple Silicon (MPS), via app.ml.utils.device.
  3. CPU — fallback if nothing else is available.

Unlike run_remote.py, this does not require the remote kernel to have this
repo checked out: the needed source files are embedded directly in the code
that gets executed remotely.

Usage:
    python -m scripts.train_dispatch
    python -m scripts.train_dispatch --no-colab   # skip the Colab check
    make train
"""

import argparse
import base64
import json
import signal
import sys
import tarfile
import uuid
from datetime import datetime
from io import BytesIO
from pathlib import Path
from types import FrameType

import requests
import websocket

from scripts.run_remote import (
    NTFY_TOPIC,
    NoRemoteKernelError,
    fetch_latest_connection,
    run_cell,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
BUNDLE_DIRS = ["app/core", "app/ml"]
CHECKPOINT_MARKER = "___TRAIN_DISPATCH_CHECKPOINT_B64___"
# Printed before CHECKPOINT_MARKER (order matters — see the extraction
# logic in dispatch_to_colab, which reads the checkpoint payload to the end
# of stdout and needs this one to end before that starts).
MLFLOW_MARKER = "___TRAIN_DISPATCH_MLFLOW_B64___"

# Errors that mean "the Colab bridge isn't reachable right now" rather than a
# bug — any of these should fall back to local training, not crash the run.
BRIDGE_UNAVAILABLE_ERRORS = (
    NoRemoteKernelError,
    requests.exceptions.RequestException,
    websocket.WebSocketException,
    OSError,
)


def _collect_sources() -> dict[str, str]:
    """Read every .py file under ``BUNDLE_DIRS``.

    Returns
    -------
    dict of str to str
        File contents keyed by repo-relative path.
    """
    files = {"app/__init__.py": (REPO_ROOT / "app" / "__init__.py").read_text()}
    for rel_dir in BUNDLE_DIRS:
        for path in (REPO_ROOT / rel_dir).rglob("*.py"):
            if "__pycache__" in path.parts:
                continue
            rel = path.relative_to(REPO_ROOT).as_posix()
            files[rel] = path.read_text()
    return files


def _build_remote_script(files: dict[str, str]) -> str:
    """Build a self-contained script to run on the remote Colab kernel.

    Parameters
    ----------
    files : dict of str to str
        Source files to embed, keyed by repo-relative path (see
        ``_collect_sources``).

    Returns
    -------
    str
        Script source that writes the bundle, trains, then prints the
        resulting checkpoint (and, best-effort, the run's
        mlflow.db/mlruns) back as base64 so the caller can save them
        locally.
    """
    payload = json.dumps(files)
    return f"""
import base64, json, subprocess, sys
from pathlib import Path

files = json.loads({payload!r})
for rel_path, content in files.items():
    p = Path(rel_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content)

subprocess.run(
    [
        sys.executable, "-m", "pip", "install", "-q",
        "pydantic>=2", "pydantic-settings", "mlflow",
    ],
    check=True,
)

sys.path.insert(0, str(Path.cwd()))
from app.ml.train.train import train  # noqa: E402

train()

# Best-effort: the checkpoint below is the critical artifact, this is a
# convenience so Colab runs still show up somewhere locally afterward.
mlflow_tar = Path("/tmp/train_dispatch_mlflow.tar.gz")
subprocess.run(
    ["tar", "czf", str(mlflow_tar), "mlflow.db", "mlruns"], check=False
)
if mlflow_tar.exists():
    print("{MLFLOW_MARKER}" + base64.b64encode(mlflow_tar.read_bytes()).decode())

checkpoint_bytes = Path("checkpoints/model_latest.pth").read_bytes()
print("{CHECKPOINT_MARKER}" + base64.b64encode(checkpoint_bytes).decode())
"""


def _raise_keyboard_interrupt(
    signum: int,
    frame: FrameType | None,  # noqa: ARG001 — required by signal.signal()'s handler signature
) -> None:
    """Signal handler that converts SIGTERM into a ``KeyboardInterrupt``.

    Parameters
    ----------
    signum : int
    frame : FrameType, optional
    """
    raise KeyboardInterrupt(f"received signal {signum}")


def _save_colab_mlflow_data(tarball_bytes: bytes) -> Path:
    """Extract a remote run's mlflow.db/mlruns into their own directory.

    Parameters
    ----------
    tarball_bytes : bytes

    Returns
    -------
    Path
        Destination directory, e.g. ``colab_runs/<timestamp>/``.

    Notes
    -----
    NOT merged into the local mlflow store, since merging two independent
    sqlite databases risks corrupting the local one. Browse it separately:
    `mlflow ui --backend-store-uri sqlite:///<dest>/mlflow.db`.
    """
    dest = REPO_ROOT / "colab_runs" / datetime.now().strftime("%Y%m%d_%H%M%S")
    dest.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=BytesIO(tarball_bytes)) as tar:
        tar.extractall(dest, filter="data")
    return dest


def dispatch_to_colab(topic: str, stale_after: int) -> bool:
    """Try to run training on a live Colab kernel.

    Parameters
    ----------
    topic : str
        ntfy.sh topic to poll for connection info.
    stale_after : int
        Passed through to ``fetch_latest_connection``.

    Returns
    -------
    bool
        Whether remote training succeeded and a checkpoint was retrieved.

    Notes
    -----
    SIGTERM's default disposition terminates the process immediately,
    bypassing `finally` — a hard kill/timeout during dispatch would silently
    leak the remote kernel (it happened once during development: an
    unattended Colab kernel kept burning GPU quota after the local script
    was killed). Converting it to KeyboardInterrupt routes it through the
    same cleanup path as Ctrl-C.
    """
    previous_sigterm = signal.signal(signal.SIGTERM, _raise_keyboard_interrupt)
    try:
        try:
            url, token = fetch_latest_connection(topic, stale_after)
            print(f"Colab GPU kernel found at {url} — dispatching training there.")

            headers = {"Authorization": f"token {token}"}
            resp = requests.post(f"{url}/api/kernels", headers=headers, timeout=15)
            kernel_id = resp.json()["id"]

            ws_url = url.replace("https://", "wss://").replace("http://", "ws://")
            ws = websocket.create_connection(
                f"{ws_url}/api/kernels/{kernel_id}/channels?token={token}", timeout=None
            )
            session_id = str(uuid.uuid4())

            try:
                script = _build_remote_script(_collect_sources())
                ok, stdout = run_cell(ws, session_id, script)
            finally:
                ws.close()
                kernel_url = f"{url}/api/kernels/{kernel_id}"
                requests.delete(kernel_url, headers=headers, timeout=15)
        except BRIDGE_UNAVAILABLE_ERRORS as e:
            print(f"Colab GPU backend unavailable ({e}); falling back to local.")
            return False
    finally:
        signal.signal(signal.SIGTERM, previous_sigterm)

    if not ok:
        print("Remote training failed; falling back to local.", file=sys.stderr)
        return False

    marker_idx = stdout.find(CHECKPOINT_MARKER)
    if marker_idx == -1:
        print(
            "Remote run finished but returned no checkpoint; "
            "falling back to local training.",
            file=sys.stderr,
        )
        return False

    b64_data = stdout[marker_idx + len(CHECKPOINT_MARKER) :].strip()
    checkpoint_bytes = base64.b64decode(b64_data)
    out_path = REPO_ROOT / "checkpoints" / "model_latest.pth"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(checkpoint_bytes)
    print(f"Checkpoint retrieved from Colab GPU and saved to {out_path}")

    mlflow_idx = stdout.find(MLFLOW_MARKER)
    if mlflow_idx != -1:
        # Printed before CHECKPOINT_MARKER, so its payload ends where that
        # marker begins rather than at the end of stdout.
        mlflow_b64 = stdout[mlflow_idx + len(MLFLOW_MARKER) : marker_idx].strip()
        try:
            dest = _save_colab_mlflow_data(base64.b64decode(mlflow_b64))
            print(
                f"Colab run's MLflow data saved to {dest} (kept separate from "
                f"the local store — view with: mlflow ui "
                f"--backend-store-uri sqlite:///{dest / 'mlflow.db'})"
            )
        except (OSError, tarfile.TarError) as e:
            print(f"Could not save Colab MLflow data ({e}); continuing anyway.")

    return True


def dispatch_local() -> None:
    """Train on the best local device (CUDA > MPS > CPU)."""
    # Deferred so a successful Colab dispatch never pays for importing torch.
    from app.ml.train.train import train  # noqa: PLC0415

    train()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--topic", default=NTFY_TOPIC, help="ntfy.sh topic to poll")
    parser.add_argument(
        "--stale-after",
        type=int,
        default=20 * 60,
        help="Treat a published kernel older than this many seconds as dead",
    )
    parser.add_argument(
        "--no-colab",
        action="store_true",
        help="Skip the Colab GPU check and train locally",
    )
    args = parser.parse_args()

    if args.no_colab or not dispatch_to_colab(args.topic, args.stale_after):
        print("Training locally (CUDA > MPS > CPU, whichever is available).")
        dispatch_local()


if __name__ == "__main__":
    main()
