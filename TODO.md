# TODO

Prioritized backlog of known gaps. This drifts from the code over time —
verify an item against the actual source before acting on it (see
`CLAUDE.md`'s "Known gaps / backlog" section for the same caveat).

## ML maturity roadmap

All six items that close the ML feedback loop (train → evaluate rigorously
→ gate promotion on evidence → validate against live traffic → monitor in
production) are done — see "Done" below, including shadow deployment and a
promotion gate that checks per-class recall and calibration, not just
aggregate accuracy. What's left in this repo is unrelated to ML maturity:
auth, GPU serving, and async batch infra (below).

## Security

- [ ] Full JWT/OAuth auth to replace the shared-secret header tokens that
      gate `POST /admin/reload-model`, optionally `POST /v1/predict`, and
      optionally `GET /metrics` (see `app/core/security.py`). All three
      tokens are opt-in and disabled by default today. This needs a real
      design decision first — the template has no user/identity model at
      all, so "OAuth" means choosing a flow (service-to-service JWT vs. a
      real user store) before writing code.

## Architecture / features

- [ ] GPU-accelerated serving — the Dockerfile builds CPU-only torch wheels
      by design; training already auto-detects CUDA/MPS locally or via the
      Colab bridge (`scripts/`).
- [ ] Redis cache / Celery workers for async batch inference —
      `POST /v1/predict/batch` (`app/api/routes/predict.py`) runs
      synchronously in-process today, bounded only by the
      `APP_MAX_CONCURRENT_PREDICTIONS` semaphore and `APP_PREDICT_RATE_LIMIT`.

## Done

- [x] **Shadow deployment before full promotion.** `InferenceService`
      (`app/services/inference_service.py`) optionally loads a second
      `model_shadow.pth` alongside `model_latest.pth`; every
      `predict_image`/`predict_batch` call also scores the shadow
      best-effort (`_score_shadow`) and records agreement/disagreement via
      `predict_shadow_agreement_total{agreement="match"|"mismatch"}` on
      `GET /metrics` — a broken or missing shadow never affects what's
      returned to a client. `make promote-shadow RUN_ID=<run-id>` stages a
      candidate (sets the registry's `"shadow"` alias, writes
      `model_shadow.pth`, no gate); `make promote-commit` promotes whatever
      is staged through the exact same gate as `make promote`
      (`commit_shadow()` in `app/ml/train/promote.py`), then clears the
      shadow slot. `POST /admin/reload-model` picks up shadow changes too
      and now reports `shadow_loaded` in its response.
- [x] **Promotion gate checks per-class recall and calibration, not just
      aggregate accuracy.** `_check_promotion_gate`
      (`app/ml/train/promote.py`) now runs three checks: accuracy
      (`APP_PROMOTION_MIN_ACCURACY_IMPROVEMENT`), per-class recall
      (`APP_PROMOTION_MAX_RECALL_REGRESSION`, default `0.05` — catches a
      candidate whose aggregate accuracy improved while one specific
      class's recall collapsed), and calibration
      (`APP_PROMOTION_MAX_CALIBRATION_REGRESSION`, default `0.05` — catches
      a model that got more accurate but also more overconfident on what
      it still gets wrong). `make train` now logs per-class recall as
      scalar `val_recall_class_N` metrics (not just inside the
      `per_class_metrics.json` artifact) so the gate can query them.
- [x] **Automated promotion gating.** `make promote` refuses a candidate
      whose `val_accuracy` regresses vs. the current champion by more than
      `APP_PROMOTION_MIN_ACCURACY_IMPROVEMENT` (default `0.0` — blocks any
      regression, allows ties), comparing MLflow-logged metrics via
      `_check_promotion_gate` (`app/ml/train/promote.py`). Skipped (with a
      warning) if there's no champion yet or either run is missing the
      metric; `--force` / `FORCE=1` (Makefile) bypasses it deliberately.
- [x] **Evaluation depth beyond a single accuracy scalar.**
      `evaluate_detailed()` / `ClassificationReport`
      (`app/ml/train/engine.py`) adds a per-class confusion matrix,
      precision/recall/F1, and Expected Calibration Error. `make train`
      logs these as MLflow artifacts/metrics once per run (not per epoch —
      the per-epoch loop still uses the cheap `evaluate()`); the
      `evaluate` CLI (`app/ml/train/evaluate.py`) now returns and prints
      the full report instead of just `val_loss`/`val_accuracy`.
- [x] **Production prediction-quality metrics.** `app/services/
      inference_service.py` now exposes `predict_confidence` (a Histogram)
      and `predict_class_total{digit="N"}` (a Counter) on every
      `predict_image`/`predict_batch` call, surfaced on the existing
      `GET /metrics` — model-quality signal alongside the HTTP-layer
      metrics `prometheus-fastapi-instrumentator` already provided.
- [x] **Version preprocessing with the weights.** `MNIST_MEAN`/`MNIST_STD`
      now live in one place (`app/ml/datasets/transform.py`), used by both
      training (`loader.py`) and serving (`inference_service.py`). Every
      checkpoint `make train` saves embeds the exact normalization used
      (`{"state_dict": ..., "normalize_mean": ..., "normalize_std": ...}`);
      `load_checkpoint()` reads it back and `InferenceService` rebuilds its
      transform from it, so a future change to the constants can't
      silently mismatch an existing checkpoint. Checkpoints saved as a
      bare state dict (pre-dating this format) fall back to today's
      defaults with a logged warning rather than breaking.
- [x] **Data versioning / fingerprinting.** `dataset_fingerprint()`
      (`app/ml/datasets/transform.py`) logs a `dataset_fingerprint` MLflow
      tag per run — a hash of the dataset identity + normalization
      constants. Not a real data hash (MNIST is a fixed, live-downloaded
      public dataset, nothing to pin yet) but establishes where one would
      plug in once this template points at a dataset that can change.
- [x] Rate limiting beyond the per-worker concurrency semaphore — added
      `slowapi` (`app/core/rate_limit.py`), applied to `/v1/predict` and
      `/v1/predict/batch` via `APP_PREDICT_RATE_LIMIT` (default
      `60/minute`, keyed by `X-API-Key` if set else remote address).
- [x] Model registry — `make train` registers every run as a version in
      the MLflow Model Registry (`registered_model_name` in
      `app/ml/train/train.py`); `make promote` points a `"champion"` alias
      at the promoted run (`app/ml/train/promote.py`) in addition to the
      existing `model_latest.pth` file copy, so "what's in production" is
      queryable from MLflow instead of only inferred from a filename.
- [x] `app/models/` empty scaffold package — removed (also dropped from
      `Makefile`'s `DIRS`/`PACKAGE_DIRS`); no concrete need for it surfaced.
- [x] Thin Ruff rule set — `pyproject.toml`'s `lint.select` now also
      includes `S`, `SIM`, `ARG`, `RET`, `RUF`, `T20`, with per-file-ignores
      for the cases where those rules don't apply (tests' stub signatures,
      scripts' intentional `print()` output, `colab_server.py`'s
      trusted-context shell calls).
