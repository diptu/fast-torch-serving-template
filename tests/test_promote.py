import uuid

import pytest
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

import app.ml.train.promote as promote_module
from app.ml.train.promote import (
    commit_shadow,
    get_champion_run_id,
    get_shadow_run_id,
    list_versions,
    promote,
    stage_shadow,
)


@pytest.fixture(autouse=True)
def _isolated_mlflow(tmp_path, monkeypatch):
    """Point every test at its own tmp sqlite store instead of the real
    project's mlflow.db — promote() now talks to the MLflow Model Registry,
    which would otherwise create/pollute the real one on every test run."""
    monkeypatch.setattr(
        promote_module.settings,
        "mlflow_tracking_uri",
        f"sqlite:///{tmp_path / 'mlflow.db'}",
    )


def _register_version(
    tmp_path,
    val_accuracy: float | None = None,
    per_class_recall: dict[int, float] | None = None,
    expected_calibration_error: float | None = None,
) -> str:
    """Register a model version under settings.mlflow_registered_model_name,
    without running a real training job. Returns the (MLflow-generated)
    run_id it's registered under."""
    client = MlflowClient(tracking_uri=f"sqlite:///{tmp_path / 'mlflow.db'}")
    name = promote_module.settings.mlflow_registered_model_name
    try:
        client.get_registered_model(name)
    except MlflowException:
        client.create_registered_model(name)
    exp_id = client.create_experiment(f"test-exp-{uuid.uuid4()}")
    run = client.create_run(experiment_id=exp_id)
    if val_accuracy is not None:
        client.log_metric(run.info.run_id, "val_accuracy", val_accuracy)
    if per_class_recall is not None:
        for cls, recall in per_class_recall.items():
            client.log_metric(run.info.run_id, f"val_recall_class_{cls}", recall)
    if expected_calibration_error is not None:
        client.log_metric(
            run.info.run_id,
            "val_expected_calibration_error",
            expected_calibration_error,
        )
    client.create_model_version(
        name, source="file:///fake-model", run_id=run.info.run_id
    )
    return run.info.run_id


def _full_recall(value: float, regress_class: int | None = None) -> dict[int, float]:
    """A recall dict covering every class settings.num_classes expects, so
    the per-class gate actually engages instead of skipping on a missing
    class. Optionally drops one class's recall to 0.5 to test a regression."""
    num_classes = promote_module.settings.num_classes
    recall = dict.fromkeys(range(num_classes), value)
    if regress_class is not None:
        recall[regress_class] = 0.5
    return recall


def test_list_versions_excludes_latest_and_sorts(tmp_path) -> None:
    (tmp_path / "model_latest.pth").write_bytes(b"x")
    (tmp_path / "model_bbb.pth").write_bytes(b"x")
    (tmp_path / "model_aaa.pth").write_bytes(b"x")

    assert list_versions(tmp_path) == ["aaa", "bbb"]


def test_list_versions_empty_dir(tmp_path) -> None:
    assert list_versions(tmp_path) == []


def test_promote_copies_versioned_checkpoint_over_latest(tmp_path) -> None:
    (tmp_path / "model_abc123.pth").write_bytes(b"trained-weights")
    (tmp_path / "model_latest.pth").write_bytes(b"stale-weights")

    target = promote("abc123", tmp_path)

    assert target == tmp_path / "model_latest.pth"
    assert target.read_bytes() == b"trained-weights"


def test_promote_exits_when_run_id_not_found(tmp_path) -> None:
    with pytest.raises(SystemExit) as exc_info:
        promote("does-not-exist", tmp_path)

    assert exc_info.value.code == 1


def test_promote_without_registered_version_warns_but_succeeds(
    tmp_path, caplog
) -> None:
    """No matching MLflow registry version (e.g. checkpoint predates this
    integration) shouldn't block the file-based promotion."""
    (tmp_path / "model_orphan.pth").write_bytes(b"trained")

    target = promote("orphan", tmp_path)

    assert target.read_bytes() == b"trained"
    assert get_champion_run_id() is None
    assert "No registered model version found" in caplog.text


def test_promote_sets_registry_champion_alias(tmp_path) -> None:
    run_id = _register_version(tmp_path)
    (tmp_path / f"model_{run_id}.pth").write_bytes(b"trained")

    promote(run_id, tmp_path)

    assert get_champion_run_id() == run_id


def test_get_champion_run_id_none_when_nothing_promoted() -> None:
    assert get_champion_run_id() is None


def test_promote_allows_first_ever_promotion_regardless_of_accuracy(
    tmp_path,
) -> None:
    run_id = _register_version(tmp_path, val_accuracy=0.10)
    (tmp_path / f"model_{run_id}.pth").write_bytes(b"trained")

    promote(run_id, tmp_path)

    assert get_champion_run_id() == run_id


def test_promote_refuses_when_candidate_regresses(tmp_path) -> None:
    champion_id = _register_version(tmp_path, val_accuracy=0.95)
    (tmp_path / f"model_{champion_id}.pth").write_bytes(b"champion-weights")
    promote(champion_id, tmp_path)

    candidate_id = _register_version(tmp_path, val_accuracy=0.80)
    (tmp_path / f"model_{candidate_id}.pth").write_bytes(b"worse-weights")

    with pytest.raises(SystemExit) as exc_info:
        promote(candidate_id, tmp_path)

    assert exc_info.value.code == 1
    assert get_champion_run_id() == champion_id
    assert (tmp_path / "model_latest.pth").read_bytes() == b"champion-weights"


def test_promote_allows_when_candidate_improves(tmp_path) -> None:
    champion_id = _register_version(tmp_path, val_accuracy=0.90)
    (tmp_path / f"model_{champion_id}.pth").write_bytes(b"old-weights")
    promote(champion_id, tmp_path)

    candidate_id = _register_version(tmp_path, val_accuracy=0.95)
    (tmp_path / f"model_{candidate_id}.pth").write_bytes(b"better-weights")

    promote(candidate_id, tmp_path)

    assert get_champion_run_id() == candidate_id
    assert (tmp_path / "model_latest.pth").read_bytes() == b"better-weights"


def test_promote_force_bypasses_regression_gate(tmp_path) -> None:
    champion_id = _register_version(tmp_path, val_accuracy=0.95)
    (tmp_path / f"model_{champion_id}.pth").write_bytes(b"champion-weights")
    promote(champion_id, tmp_path)

    candidate_id = _register_version(tmp_path, val_accuracy=0.10)
    (tmp_path / f"model_{candidate_id}.pth").write_bytes(b"forced-weights")

    promote(candidate_id, tmp_path, force=True)

    assert get_champion_run_id() == candidate_id
    assert (tmp_path / "model_latest.pth").read_bytes() == b"forced-weights"


def test_promote_allows_when_val_accuracy_missing(tmp_path, caplog) -> None:
    champion_id = _register_version(tmp_path, val_accuracy=0.95)
    (tmp_path / f"model_{champion_id}.pth").write_bytes(b"champion-weights")
    promote(champion_id, tmp_path)

    candidate_id = _register_version(tmp_path)  # no val_accuracy logged
    (tmp_path / f"model_{candidate_id}.pth").write_bytes(b"no-metric-weights")

    promote(candidate_id, tmp_path)

    assert get_champion_run_id() == candidate_id
    assert "Skipping accuracy gate" in caplog.text


def test_promote_refuses_when_per_class_recall_regresses(tmp_path) -> None:
    champion_id = _register_version(
        tmp_path, val_accuracy=0.90, per_class_recall=_full_recall(0.95)
    )
    (tmp_path / f"model_{champion_id}.pth").write_bytes(b"champion-weights")
    promote(champion_id, tmp_path)

    # Aggregate accuracy improves, but class 4's recall collapses.
    candidate_id = _register_version(
        tmp_path,
        val_accuracy=0.95,
        per_class_recall=_full_recall(0.95, regress_class=4),
    )
    (tmp_path / f"model_{candidate_id}.pth").write_bytes(b"worse-class-weights")

    with pytest.raises(SystemExit) as exc_info:
        promote(candidate_id, tmp_path)

    assert exc_info.value.code == 1
    assert get_champion_run_id() == champion_id


def test_promote_allows_per_class_recall_within_tolerance(tmp_path) -> None:
    champion_id = _register_version(
        tmp_path, val_accuracy=0.90, per_class_recall=_full_recall(0.90)
    )
    (tmp_path / f"model_{champion_id}.pth").write_bytes(b"champion-weights")
    promote(champion_id, tmp_path)

    # 0.90 -> 0.87 is within the default 0.05 tolerance.
    recall = _full_recall(0.90)
    recall[3] = 0.87
    candidate_id = _register_version(
        tmp_path, val_accuracy=0.91, per_class_recall=recall
    )
    (tmp_path / f"model_{candidate_id}.pth").write_bytes(b"fine-weights")

    promote(candidate_id, tmp_path)

    assert get_champion_run_id() == candidate_id


def test_promote_allows_when_per_class_recall_missing(tmp_path, caplog) -> None:
    champion_id = _register_version(tmp_path, val_accuracy=0.90)
    (tmp_path / f"model_{champion_id}.pth").write_bytes(b"champion-weights")
    promote(champion_id, tmp_path)

    candidate_id = _register_version(tmp_path, val_accuracy=0.95)
    (tmp_path / f"model_{candidate_id}.pth").write_bytes(b"no-recall-weights")

    promote(candidate_id, tmp_path)

    assert get_champion_run_id() == candidate_id
    assert "Skipping per-class recall gate" in caplog.text


def test_promote_refuses_when_calibration_regresses(tmp_path) -> None:
    champion_id = _register_version(
        tmp_path, val_accuracy=0.90, expected_calibration_error=0.02
    )
    (tmp_path / f"model_{champion_id}.pth").write_bytes(b"champion-weights")
    promote(champion_id, tmp_path)

    candidate_id = _register_version(
        tmp_path, val_accuracy=0.95, expected_calibration_error=0.20
    )
    (tmp_path / f"model_{candidate_id}.pth").write_bytes(b"overconfident-weights")

    with pytest.raises(SystemExit) as exc_info:
        promote(candidate_id, tmp_path)

    assert exc_info.value.code == 1
    assert get_champion_run_id() == champion_id


def test_promote_allows_calibration_within_tolerance(tmp_path) -> None:
    champion_id = _register_version(
        tmp_path, val_accuracy=0.90, expected_calibration_error=0.02
    )
    (tmp_path / f"model_{champion_id}.pth").write_bytes(b"champion-weights")
    promote(champion_id, tmp_path)

    # 0.02 -> 0.06 is within the default 0.05 tolerance.
    candidate_id = _register_version(
        tmp_path, val_accuracy=0.95, expected_calibration_error=0.06
    )
    (tmp_path / f"model_{candidate_id}.pth").write_bytes(b"fine-weights")

    promote(candidate_id, tmp_path)

    assert get_champion_run_id() == candidate_id


def test_promote_allows_when_calibration_missing(tmp_path, caplog) -> None:
    champion_id = _register_version(tmp_path, val_accuracy=0.90)
    (tmp_path / f"model_{champion_id}.pth").write_bytes(b"champion-weights")
    promote(champion_id, tmp_path)

    candidate_id = _register_version(tmp_path, val_accuracy=0.95)
    (tmp_path / f"model_{candidate_id}.pth").write_bytes(b"no-ece-weights")

    promote(candidate_id, tmp_path)

    assert get_champion_run_id() == candidate_id
    assert "Skipping calibration gate" in caplog.text


def test_list_versions_excludes_shadow_file(tmp_path) -> None:
    (tmp_path / "model_latest.pth").write_bytes(b"x")
    (tmp_path / "model_shadow.pth").write_bytes(b"x")
    (tmp_path / "model_aaa.pth").write_bytes(b"x")

    assert list_versions(tmp_path) == ["aaa"]


def test_stage_shadow_writes_file_and_sets_shadow_alias(tmp_path) -> None:
    run_id = _register_version(tmp_path)
    (tmp_path / f"model_{run_id}.pth").write_bytes(b"candidate-weights")

    target = stage_shadow(run_id, tmp_path)

    assert target == tmp_path / "model_shadow.pth"
    assert target.read_bytes() == b"candidate-weights"
    assert get_shadow_run_id() == run_id
    assert get_champion_run_id() is None  # staging isn't promoting


def test_stage_shadow_exits_when_run_id_not_found(tmp_path) -> None:
    with pytest.raises(SystemExit) as exc_info:
        stage_shadow("does-not-exist", tmp_path)

    assert exc_info.value.code == 1


def test_get_shadow_run_id_none_when_nothing_staged() -> None:
    assert get_shadow_run_id() is None


def test_commit_shadow_promotes_staged_run_and_clears_shadow_file(tmp_path) -> None:
    run_id = _register_version(tmp_path, val_accuracy=0.95)
    (tmp_path / f"model_{run_id}.pth").write_bytes(b"candidate-weights")
    stage_shadow(run_id, tmp_path)

    target = commit_shadow(tmp_path)

    assert target == tmp_path / "model_latest.pth"
    assert target.read_bytes() == b"candidate-weights"
    assert get_champion_run_id() == run_id
    assert not (tmp_path / "model_shadow.pth").exists()


def test_commit_shadow_exits_when_nothing_staged(tmp_path) -> None:
    with pytest.raises(SystemExit) as exc_info:
        commit_shadow(tmp_path)

    assert exc_info.value.code == 1


def test_commit_shadow_runs_promotion_gate(tmp_path) -> None:
    champion_id = _register_version(tmp_path, val_accuracy=0.95)
    (tmp_path / f"model_{champion_id}.pth").write_bytes(b"champion-weights")
    promote(champion_id, tmp_path)

    shadow_id = _register_version(tmp_path, val_accuracy=0.10)
    (tmp_path / f"model_{shadow_id}.pth").write_bytes(b"bad-shadow-weights")
    stage_shadow(shadow_id, tmp_path)

    with pytest.raises(SystemExit) as exc_info:
        commit_shadow(tmp_path)

    assert exc_info.value.code == 1
    assert get_champion_run_id() == champion_id
    # A refused commit doesn't clear the shadow slot — nothing was committed.
    assert get_shadow_run_id() == shadow_id


def test_commit_shadow_force_bypasses_gate(tmp_path) -> None:
    champion_id = _register_version(tmp_path, val_accuracy=0.95)
    (tmp_path / f"model_{champion_id}.pth").write_bytes(b"champion-weights")
    promote(champion_id, tmp_path)

    shadow_id = _register_version(tmp_path, val_accuracy=0.10)
    (tmp_path / f"model_{shadow_id}.pth").write_bytes(b"forced-shadow-weights")
    stage_shadow(shadow_id, tmp_path)

    commit_shadow(tmp_path, force=True)

    assert get_champion_run_id() == shadow_id
    assert not (tmp_path / "model_shadow.pth").exists()


def test_main_list_prints_available_versions(tmp_path, monkeypatch, capsys) -> None:
    (tmp_path / "model_run1.pth").write_bytes(b"x")
    monkeypatch.setattr(promote_module.settings, "checkpoint_dir", tmp_path)
    monkeypatch.setattr("sys.argv", ["promote.py", "--list"])

    promote_module.main()

    assert capsys.readouterr().out.strip() == "run1"


def test_main_list_marks_champion(tmp_path, monkeypatch, capsys) -> None:
    run_id = _register_version(tmp_path)
    (tmp_path / f"model_{run_id}.pth").write_bytes(b"x")
    other_id = "z" * len(run_id)
    (tmp_path / f"model_{other_id}.pth").write_bytes(b"x")
    monkeypatch.setattr(promote_module.settings, "checkpoint_dir", tmp_path)
    promote(run_id, tmp_path)
    monkeypatch.setattr("sys.argv", ["promote.py", "--list"])

    promote_module.main()

    # run_id is all lowercase hex, which sorts before "z" alphabetically.
    lines = capsys.readouterr().out.strip().splitlines()
    assert lines == [f"{run_id} (champion)", other_id]


def test_main_list_marks_shadow_and_champion_separately(
    tmp_path, monkeypatch, capsys
) -> None:
    champion_id = _register_version(tmp_path)
    (tmp_path / f"model_{champion_id}.pth").write_bytes(b"x")
    shadow_id = _register_version(tmp_path)
    (tmp_path / f"model_{shadow_id}.pth").write_bytes(b"y")
    monkeypatch.setattr(promote_module.settings, "checkpoint_dir", tmp_path)
    promote(champion_id, tmp_path)
    stage_shadow(shadow_id, tmp_path)
    monkeypatch.setattr("sys.argv", ["promote.py", "--list"])

    promote_module.main()

    lines = set(capsys.readouterr().out.strip().splitlines())
    assert lines == {f"{champion_id} (champion)", f"{shadow_id} (shadow)"}


def test_main_list_reports_when_nothing_to_promote(
    tmp_path, monkeypatch, capsys
) -> None:
    monkeypatch.setattr(promote_module.settings, "checkpoint_dir", tmp_path)
    monkeypatch.setattr("sys.argv", ["promote.py", "--list"])

    promote_module.main()

    assert "No versioned checkpoints found." in capsys.readouterr().out


def test_main_promotes_given_run_id(tmp_path, monkeypatch) -> None:
    (tmp_path / "model_run1.pth").write_bytes(b"trained")
    monkeypatch.setattr(promote_module.settings, "checkpoint_dir", tmp_path)
    monkeypatch.setattr("sys.argv", ["promote.py", "--run-id", "run1"])

    promote_module.main()

    assert (tmp_path / "model_latest.pth").read_bytes() == b"trained"


def test_main_requires_run_id_without_list(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(promote_module.settings, "checkpoint_dir", tmp_path)
    monkeypatch.setattr("sys.argv", ["promote.py"])

    with pytest.raises(SystemExit) as exc_info:
        promote_module.main()

    assert exc_info.value.code == 2


def test_main_shadow_flag_stages_run(tmp_path, monkeypatch) -> None:
    (tmp_path / "model_run1.pth").write_bytes(b"candidate")
    monkeypatch.setattr(promote_module.settings, "checkpoint_dir", tmp_path)
    monkeypatch.setattr("sys.argv", ["promote.py", "--shadow", "run1"])

    promote_module.main()

    assert (tmp_path / "model_shadow.pth").read_bytes() == b"candidate"
    assert not (tmp_path / "model_latest.pth").exists()


def test_main_commit_flag_commits_staged_shadow(tmp_path, monkeypatch) -> None:
    run_id = _register_version(tmp_path, val_accuracy=0.95)
    (tmp_path / f"model_{run_id}.pth").write_bytes(b"candidate")
    monkeypatch.setattr(promote_module.settings, "checkpoint_dir", tmp_path)
    stage_shadow(run_id, tmp_path)
    monkeypatch.setattr("sys.argv", ["promote.py", "--commit"])

    promote_module.main()

    assert (tmp_path / "model_latest.pth").read_bytes() == b"candidate"
    assert get_champion_run_id() == run_id
