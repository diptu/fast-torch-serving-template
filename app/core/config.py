import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    YamlConfigSettingsSource,
)


class Settings(BaseSettings):
    """Application configuration, env prefix ``APP_``.

    Sourced (highest priority first) from explicit kwargs, environment
    variables, ``.env``, ``configs/default.yaml``, then field defaults —
    see ``settings_customise_sources``. Access via the cached
    ``get_settings()``, not by instantiating this directly.
    """

    model_config = SettingsConfigDict(
        env_prefix="APP_",
        env_file=".env",
        extra="ignore",
        yaml_file="configs/default.yaml",
    )

    # Deployment environment. Only effect today: gates /docs, /redoc, and
    # /openapi.json off in "production" (see app/main.py) so the schema
    # isn't publicly discoverable. Defaults to "development" so local/dev
    # usage keeps docs on without any config.
    environment: Literal["development", "staging", "production"] = "development"

    # Passed straight to logging.Logger.setLevel via setup_logging(). "INFO"
    # is the right default for production; set APP_LOG_LEVEL=DEBUG locally
    # when you need per-request detail beyond what's normally logged.
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"

    # Paths
    data_dir: Path = Field(default=Path("data"))
    checkpoint_dir: Path = Field(default=Path("checkpoints"))

    # ML Configuration
    mlflow_tracking_uri: str = "sqlite:///mlflow.db"
    mlflow_experiment_name: str = "mnist-cnn"
    # Name model versions register under in the MLflow Model Registry —
    # distinct from mlflow_experiment_name, which just groups runs. Every
    # training run registers a new version under this name (app/ml/train/
    # train.py); `make promote` points a "champion" alias at one of them
    # (app/ml/train/promote.py).
    mlflow_registered_model_name: str = "mnist-cnn"
    # `make promote` refuses a candidate whose val_accuracy is more than
    # this far below the current champion's — 0.0 blocks any regression
    # while still allowing ties; raise it to require strict improvement, or
    # override a specific promotion with --force. Skipped entirely (with a
    # warning) if either run has no val_accuracy logged, or there's no
    # champion yet to compare against.
    promotion_min_accuracy_improvement: float = 0.0
    # Per-class recall is noisier than aggregate accuracy on a fixed-size
    # validation set, so it gets its own (looser) tolerance rather than
    # reusing promotion_min_accuracy_improvement — an aggregate-accuracy
    # win that quietly craters one class's recall still gets refused.
    promotion_max_recall_regression: float = 0.05
    # How much worse (higher) expected_calibration_error is allowed to get.
    # A model can improve accuracy while becoming meaningfully more
    # overconfident on what it still gets wrong; this catches that too.
    promotion_max_calibration_regression: float = 0.05
    num_classes: int = Field(default=10, gt=0)
    SEED: int = Field(default=42, gt=0)

    # Model Hyperparameters
    BATCH_SIZE: int = Field(default=64, gt=0)
    LEARNING_RATE: float = Field(default=0.001, gt=0)
    EPOCHS: int = Field(default=10, gt=0)
    # Device validation ensures we only use supported hardware strings
    device: Literal["cpu", "cuda", "mps"] = "cpu"

    # Admin API (e.g. POST /admin/reload-model). Empty disables it entirely —
    # there's no broader auth system yet, so this is opt-in shared-secret
    # protection rather than an endpoint anyone can hit.
    admin_token: str = ""

    # Optional shared-secret protection for /v1/predict itself (X-API-Key
    # header). Empty (the default) means anyone can call it, unchanged from
    # before this setting existed — set this before exposing the API
    # publicly if that's not what you want.
    predict_api_key: str = ""

    # Optional shared-secret protection for GET /metrics (X-Metrics-Token
    # header). Empty (the default) leaves it open, same as before this
    # setting existed — /metrics has no other auth gate, unlike /admin/* and
    # optionally /v1/predict, so set this before exposing the API publicly.
    metrics_token: str = ""

    # No origins allowed by default — this is an API template with no known
    # frontend, so "locked down until configured" is the safe default. Set
    # e.g. APP_CORS_ALLOW_ORIGINS='["http://localhost:3000"]' to open it up.
    cors_allow_origins: list[str] = Field(default_factory=list)

    # Bounds how many predictions run at once per worker process, so a burst
    # of uploads can't pile up unbounded CPU/memory pressure. Defaults to
    # core count since inference is CPU-bound. A request that can't get a
    # slot within prediction_queue_timeout_seconds gets a 503 instead of
    # queueing indefinitely.
    max_concurrent_predictions: int = Field(
        default_factory=lambda: os.cpu_count() or 4, gt=0
    )
    prediction_queue_timeout_seconds: float = Field(default=5.0, gt=0)

    # Per-client request rate on /v1/predict and /v1/predict/batch (slowapi
    # syntax, e.g. "60/minute"), keyed by X-API-Key if predict_api_key is
    # set, else by remote address. Unlike the semaphore above (which bounds
    # concurrency), this bounds total request *rate* — on by default with a
    # generous limit, since (unlike a secret) there's a safe default here.
    # Set to "" to disable.
    predict_rate_limit: str = "60/minute"

    # Distributed tracing (OpenTelemetry). Empty (the default) means tracing
    # is entirely off — no collector to send to, no infra needed. Set this
    # to an OTLP HTTP endpoint (e.g. http://localhost:4318) to enable it.
    otel_exporter_otlp_endpoint: str = ""
    otel_service_name: str = "fast-torch-serving-template"

    # Git commit this image was built from, surfaced via GET /version.
    # "unknown" (the default) is what you get running outside the Docker
    # image — the Dockerfile sets this from a build arg (see CD workflow),
    # since there's no .git directory in the runtime image to read it from.
    git_sha: str = "unknown"

    @field_validator("data_dir", "checkpoint_dir")
    @classmethod
    def check_directories(cls, v: Path) -> Path:
        """Ensure a configured directory exists, creating it if needed.

        Parameters
        ----------
        v : Path

        Returns
        -------
        Path
            ``v``, unchanged.
        """
        v.mkdir(parents=True, exist_ok=True)
        return v

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        """Set source priority: kwargs > env > .env > YAML > secrets file.

        Returns
        -------
        tuple of PydanticBaseSettingsSource
            The YAML file is optional — missing it just falls through to
            field defaults.
        """
        return (
            init_settings,
            env_settings,
            dotenv_settings,
            YamlConfigSettingsSource(settings_cls),
            file_secret_settings,
        )


@lru_cache
def get_settings() -> Settings:
    """Get the process-wide cached ``Settings`` singleton.

    Returns
    -------
    Settings
    """
    return Settings()
