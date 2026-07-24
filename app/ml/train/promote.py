"""Promote a versioned training run to be the model the API serves.

Training (`app/ml/train/train.py`) saves both `model_latest.pth` (what
InferenceService loads) and a run-tagged `model_<run_id>.pth`. This formalizes
"roll back to a known-good run" into a real command instead of a manual file
copy. It also points the "champion" alias in the MLflow Model Registry
(settings.mlflow_registered_model_name) at whichever run gets promoted, so
"what's currently in production" is queryable from MLflow, not just inferred
from which local file happens to be named model_latest.pth.

Before promoting, `_check_promotion_gate` refuses a candidate that regresses
vs. the champion on aggregate val_accuracy, any single class's recall, or
calibration (expected_calibration_error) — an aggregate-accuracy win can
still hide a collapsed class or a model that's become badly overconfident.

`stage_shadow`/`commit_shadow` let a candidate run be validated against live
traffic (see InferenceService's shadow scoring) before it's promoted at all:
`model_shadow.pth` is loaded alongside `model_latest.pth` and scored on every
request without ever affecting what's returned, so its live agreement rate
with the champion is known before `--commit` runs it through the same
promotion gate as a normal `promote()`.
"""

import argparse
import shutil
import sys
from pathlib import Path

from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

from app.core.config import get_settings
from app.core.logging import get_logger, setup_logging

settings = get_settings()
logger = get_logger(__name__)

_PREFIX = "model_"
_SUFFIX = ".pth"
_CHAMPION_ALIAS = "champion"
_SHADOW_ALIAS = "shadow"


def _registry_client() -> MlflowClient:
    return MlflowClient(tracking_uri=settings.mlflow_tracking_uri)


def _set_registry_alias(run_id: str, alias: str) -> None:
    """Point a registry alias at whichever version this run produced.

    Parameters
    ----------
    run_id : str
    alias : str
        ``_CHAMPION_ALIAS`` or ``_SHADOW_ALIAS``.

    Notes
    -----
    Best-effort: the local checkpoint copy in ``promote()``/``stage_shadow()``
    is what actually changes what the API serves, so a run with no matching
    registered version (e.g. trained before this integration existed, or
    trained without ``registered_model_name`` set) is logged and skipped
    rather than failing the whole command.
    """
    client = _registry_client()
    name = settings.mlflow_registered_model_name
    versions = client.search_model_versions(f"name='{name}' and run_id='{run_id}'")
    if not versions:
        logger.warning(
            f"No registered model version found for run {run_id} under "
            f"'{name}' — skipping registry alias update."
        )
        return

    version = versions[0].version
    client.set_registered_model_alias(name, alias, version)
    logger.info(f"Registry alias '{alias}' -> {name} v{version} (run {run_id})")


def _get_alias_run_id(alias: str) -> str | None:
    """Look up which run_id a registry alias currently points to.

    Parameters
    ----------
    alias : str

    Returns
    -------
    str, optional
        None if the alias isn't set, or the registered model doesn't exist
        at all.
    """
    try:
        version = _registry_client().get_model_version_by_alias(
            settings.mlflow_registered_model_name, alias
        )
    except MlflowException:
        return None
    return version.run_id


def get_champion_run_id() -> str | None:
    """Look up which run_id the registry's "champion" alias currently points to.

    Returns
    -------
    str, optional
        None if nothing has been promoted through the registry yet.
    """
    return _get_alias_run_id(_CHAMPION_ALIAS)


def get_shadow_run_id() -> str | None:
    """Look up which run_id the registry's "shadow" alias currently points to.

    Returns
    -------
    str, optional
        None if nothing is currently staged as a shadow.
    """
    return _get_alias_run_id(_SHADOW_ALIAS)


def _get_metric(run_id: str, metric_name: str) -> float | None:
    """Fetch a single scalar metric's logged value from a run.

    Parameters
    ----------
    run_id : str
    metric_name : str

    Returns
    -------
    float, optional
        None if the run doesn't exist or never logged ``metric_name``.
    """
    try:
        run = _registry_client().get_run(run_id)
    except MlflowException:
        return None
    value = run.data.metrics.get(metric_name)
    return float(value) if value is not None else None


def _check_accuracy_gate(run_id: str, champion_run_id: str) -> None:
    """Refuse promotion if aggregate val_accuracy regresses vs. the champion.

    Parameters
    ----------
    run_id : str
    champion_run_id : str

    Raises
    ------
    SystemExit
        If both runs have a logged ``val_accuracy`` and the candidate's is
        below the champion's by more than
        ``settings.promotion_min_accuracy_improvement``.
    """
    candidate_acc = _get_metric(run_id, "val_accuracy")
    champion_acc = _get_metric(champion_run_id, "val_accuracy")
    if candidate_acc is None or champion_acc is None:
        logger.warning(
            "Skipping accuracy gate: val_accuracy missing for "
            f"candidate ({candidate_acc}) or champion ({champion_acc})."
        )
        return

    required = champion_acc + settings.promotion_min_accuracy_improvement
    if candidate_acc < required:
        logger.error(
            f"Refusing to promote {run_id}: val_accuracy {candidate_acc:.4f} "
            f"is below the required {required:.4f} (champion {champion_run_id} "
            f"is at {champion_acc:.4f}). Use --force to override."
        )
        sys.exit(1)


def _check_per_class_recall_gate(run_id: str, champion_run_id: str) -> None:
    """Refuse promotion if any single class's recall regresses vs. the champion.

    Parameters
    ----------
    run_id : str
    champion_run_id : str

    Raises
    ------
    SystemExit
        If, for some class, both runs have a logged ``val_recall_class_N``
        and the candidate's is below the champion's by more than
        ``settings.promotion_max_recall_regression``.

    Notes
    -----
    Catches what the aggregate accuracy gate can't: a candidate whose
    overall accuracy improved while one specific class's recall collapsed.
    Skips the whole check (not just one class) the first time either run is
    missing a class's metric — older runs logged before ``val_recall_class_N``
    existed have none of these, so a partial skip would be misleading.
    """
    for cls in range(settings.num_classes):
        metric_name = f"val_recall_class_{cls}"
        candidate_recall = _get_metric(run_id, metric_name)
        champion_recall = _get_metric(champion_run_id, metric_name)
        if candidate_recall is None or champion_recall is None:
            logger.warning(
                f"Skipping per-class recall gate: {metric_name} missing for "
                f"candidate ({candidate_recall}) or champion ({champion_recall})."
            )
            return

        required = champion_recall - settings.promotion_max_recall_regression
        if candidate_recall < required:
            logger.error(
                f"Refusing to promote {run_id}: class {cls} recall "
                f"{candidate_recall:.4f} is below the required {required:.4f} "
                f"(champion {champion_run_id} is at {champion_recall:.4f}). "
                "Use --force to override."
            )
            sys.exit(1)


def _check_calibration_gate(run_id: str, champion_run_id: str) -> None:
    """Refuse promotion if calibration error regresses vs. the champion.

    Parameters
    ----------
    run_id : str
    champion_run_id : str

    Raises
    ------
    SystemExit
        If both runs have a logged ``val_expected_calibration_error`` and
        the candidate's is above the champion's by more than
        ``settings.promotion_max_calibration_regression``.

    Notes
    -----
    A model can improve accuracy while becoming meaningfully more
    overconfident on what it still gets wrong; the accuracy gate alone
    can't see that.
    """
    candidate_ece = _get_metric(run_id, "val_expected_calibration_error")
    champion_ece = _get_metric(champion_run_id, "val_expected_calibration_error")
    if candidate_ece is None or champion_ece is None:
        logger.warning(
            "Skipping calibration gate: val_expected_calibration_error "
            f"missing for candidate ({candidate_ece}) or champion ({champion_ece})."
        )
        return

    allowed = champion_ece + settings.promotion_max_calibration_regression
    if candidate_ece > allowed:
        logger.error(
            f"Refusing to promote {run_id}: calibration error {candidate_ece:.4f} "
            f"exceeds the allowed {allowed:.4f} (champion {champion_run_id} is at "
            f"{champion_ece:.4f}). Use --force to override."
        )
        sys.exit(1)


def _check_promotion_gate(run_id: str) -> None:
    """Run every promotion gate check against the current champion.

    Parameters
    ----------
    run_id : str

    Raises
    ------
    SystemExit
        See ``_check_accuracy_gate``, ``_check_per_class_recall_gate``, and
        ``_check_calibration_gate``.

    Notes
    -----
    Allows the promotion through entirely (with a warning, not a failure)
    when there's no champion yet — since this gate should block *known*
    regressions, not invent risk where there's no data to judge by. Bypass
    everything with ``promote(..., force=True)`` / ``--force`` for
    deliberate overrides, e.g. rolling back to an older run for reasons
    these metrics don't capture.
    """
    champion_run_id = get_champion_run_id()
    if champion_run_id is None:
        return

    _check_accuracy_gate(run_id, champion_run_id)
    _check_per_class_recall_gate(run_id, champion_run_id)
    _check_calibration_gate(run_id, champion_run_id)


def list_versions(checkpoint_dir: Path) -> list[str]:
    """List run IDs of every versioned checkpoint available to promote.

    Parameters
    ----------
    checkpoint_dir : Path

    Returns
    -------
    list of str
        Sorted run IDs (excludes ``model_latest.pth``/``model_shadow.pth``).
    """
    excluded = {"model_latest.pth", "model_shadow.pth"}
    return sorted(
        p.name[len(_PREFIX) : -len(_SUFFIX)]
        for p in checkpoint_dir.glob(f"{_PREFIX}*{_SUFFIX}")
        if p.name not in excluded
    )


def promote(run_id: str, checkpoint_dir: Path, *, force: bool = False) -> Path:
    """Copy ``model_<run_id>.pth`` over ``model_latest.pth``.

    Parameters
    ----------
    run_id : str
    checkpoint_dir : Path
    force : bool, default False
        Skip ``_check_promotion_gate`` (the accuracy/recall/calibration
        regression checks).

    Returns
    -------
    Path
        The updated ``model_latest.pth`` path.

    Raises
    ------
    SystemExit
        If no checkpoint exists for ``run_id``, or the promotion gate
        refuses it (see ``_check_promotion_gate``).

    Notes
    -----
    Also best-effort sets the registry "champion" alias to this run (see
    ``_set_registry_alias``). Follow up with `POST /admin/reload-model` (or
    a restart) to pick the file change up without a redeploy.
    """
    source = checkpoint_dir / f"{_PREFIX}{run_id}{_SUFFIX}"
    if not source.exists():
        available = list_versions(checkpoint_dir)
        logger.error(
            f"No checkpoint found at {source}. "
            f"Available run IDs: {', '.join(available) or '(none)'}"
        )
        sys.exit(1)

    if not force:
        _check_promotion_gate(run_id)

    target = checkpoint_dir / f"model_latest{_SUFFIX}"
    shutil.copyfile(source, target)
    logger.info(f"Promoted {source.name} -> {target.name}")
    _set_registry_alias(run_id, _CHAMPION_ALIAS)
    return target


def stage_shadow(run_id: str, checkpoint_dir: Path) -> Path:
    """Stage a versioned run as the shadow model, without promoting it.

    Parameters
    ----------
    run_id : str
    checkpoint_dir : Path

    Returns
    -------
    Path
        The written ``model_shadow.pth`` path.

    Raises
    ------
    SystemExit
        If no checkpoint exists for ``run_id``.

    Notes
    -----
    Unlike ``promote()``, this never touches ``model_latest.pth`` and never
    runs the promotion gate — staging a shadow is explicitly lower-risk (it
    can't change what real traffic gets served, see InferenceService), so
    there's nothing to gate yet. Call ``POST /admin/reload-model`` (or
    restart) for a running InferenceService to pick it up and start
    shadow-scoring live traffic; use ``commit_shadow`` once its agreement
    rate looks acceptable.
    """
    source = checkpoint_dir / f"{_PREFIX}{run_id}{_SUFFIX}"
    if not source.exists():
        available = list_versions(checkpoint_dir)
        logger.error(
            f"No checkpoint found at {source}. "
            f"Available run IDs: {', '.join(available) or '(none)'}"
        )
        sys.exit(1)

    target = checkpoint_dir / f"model_shadow{_SUFFIX}"
    shutil.copyfile(source, target)
    logger.info(f"Staged {source.name} -> {target.name}")
    _set_registry_alias(run_id, _SHADOW_ALIAS)
    return target


def commit_shadow(checkpoint_dir: Path, *, force: bool = False) -> Path:
    """Promote whatever run is currently staged as the shadow to champion.

    Parameters
    ----------
    checkpoint_dir : Path
    force : bool, default False
        Passed through to ``promote()``.

    Returns
    -------
    Path
        The updated ``model_latest.pth`` path.

    Raises
    ------
    SystemExit
        If nothing is currently staged as a shadow, or ``promote()`` refuses
        it (same promotion gate as any other candidate).

    Notes
    -----
    Runs the shadow run through the exact same ``promote()`` (and its
    promotion gate) as any other candidate — shadow scoring de-risks a
    promotion, it doesn't replace this gate. Clears ``model_shadow.pth``
    afterward: once committed, there's no longer an active shadow
    experiment in progress.
    """
    shadow_run_id = get_shadow_run_id()
    if shadow_run_id is None:
        logger.error("No shadow currently staged — use --shadow=<run-id> first.")
        sys.exit(1)

    target = promote(shadow_run_id, checkpoint_dir, force=force)

    shadow_path = checkpoint_dir / f"model_shadow{_SUFFIX}"
    shadow_path.unlink(missing_ok=True)
    logger.info(
        f"Committed shadow run {shadow_run_id} to champion; cleared shadow slot."
    )
    return target


def main() -> None:
    setup_logging(settings.log_level)
    parser = argparse.ArgumentParser(
        description="Promote a versioned training run to model_latest.pth."
    )
    parser.add_argument("--run-id", help="Run ID to promote (see --list)")
    parser.add_argument(
        "--list", action="store_true", help="List available run IDs and exit"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Skip the promotion gate (accuracy/recall/calibration regression checks)",
    )
    parser.add_argument(
        "--shadow", metavar="RUN_ID", help="Stage RUN_ID as the shadow model"
    )
    parser.add_argument(
        "--commit",
        action="store_true",
        help="Promote the currently staged shadow to champion",
    )
    args = parser.parse_args()

    if args.list:
        available = list_versions(settings.checkpoint_dir)
        champion = get_champion_run_id()
        shadow = get_shadow_run_id()
        lines = []
        for v in available:
            tags = [
                tag
                for tag, run_id in (("champion", champion), ("shadow", shadow))
                if v == run_id
            ]
            lines.append(f"{v} ({', '.join(tags)})" if tags else v)
        print("\n".join(lines) if lines else "No versioned checkpoints found.")
        return

    if args.commit:
        commit_shadow(settings.checkpoint_dir, force=args.force)
        return

    if args.shadow:
        stage_shadow(args.shadow, settings.checkpoint_dir)
        return

    if not args.run_id:
        parser.error("--run-id is required (or use --list/--shadow/--commit instead)")

    promote(args.run_id, settings.checkpoint_dir, force=args.force)


if __name__ == "__main__":
    main()
