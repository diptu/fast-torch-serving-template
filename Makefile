# Unified Makefile for Production-Ready Development
# Features: Scaffolding, Dependency Management, Linting, Security, Testing, and Cleanup

.PHONY: help init scaffold lint check test clean purge train evaluate mlflow-ui \
        docker-build docker-run docker-stop docker-logs promote promote-list \
        promote-shadow promote-commit export-onnx

# Running bare `make` (or `make help`) lists every target below with its
# description instead of silently doing something — so nobody has to
# memorize target names/flags, just read this list. Keep every target's
# `## comment` in sync with what it actually does.
.DEFAULT_GOAL := help

# --- Configuration ---
# Uses 'uv' to manage the environment and dependencies
PYTHON := uv run python
IMAGE_NAME := fast-torch-serving
CONTAINER_NAME := fast-torch-serving
WEB_CONCURRENCY := 4

# Define the project structure
DIRS := app/api/routes app/core app/schemas app/services app/ml/models app/ml/inference app/ml/datasets app/ml/utils app/utils tests docker scripts .github/workflows checkpoints mlruns
FILES := app/api/dependencies.py app/core/config.py app/core/logging.py app/core/security.py app/main.py scripts/colab_server.py scripts/gpu_sanity_check.py scripts/run_remote.py pyproject.toml Dockerfile docker-compose.yml README.md
PACKAGE_DIRS := app app/api app/api/routes app/core app/schemas app/services app/ml app/ml/models app/ml/inference app/ml/datasets app/ml/utils app/utils scripts

# --- Targets ---

# 0. Help: List every target below with its description. This is the
# default goal — running bare `make` shows this instead of doing anything.
help: ## Show this list of commands
	@echo "Usage: make <target> [VAR=value ...]"
	@echo
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'
	@echo
	@echo "Variables: RUN_ID + optional FORCE=1 (for promote), WEB_CONCURRENCY"
	@echo "(for docker-run, currently $(WEB_CONCURRENCY)) — e.g. make docker-run WEB_CONCURRENCY=8"

# 1. Scaffolding: Create the foundational directory structure
scaffold: ## Create the project's directory/file skeleton
	@echo "Creating directory structure..."
	@mkdir -p $(DIRS)
	@echo "Creating placeholder files..."
	@touch $(FILES)
	@echo "Creating __init__.py files..."
	@for dir in $(PACKAGE_DIRS); do touch $$dir/__init__.py; done
	@echo "Scaffold complete."

# 2. Initialization: Sync environment and install pre-commit hooks
init: scaffold ## Sync deps (incl. train group) and install pre-commit hooks
	@echo "Syncing dependencies and installing git hooks..."
	@uv sync --group train
	@uv run pre-commit install
	@echo "Environment initialized and hooks active."

# 3. Quality Control: Linting and Formatting
lint: ## Ruff check --fix + format
	@echo "Running Ruff (formatting and linting)..."
	@uv run ruff check . --fix
	@uv run ruff format .

# 4. Verification: Static analysis, Type checking, and Security
check: ## mypy --strict + bandit security audit
	@echo "Running Type Checking (mypy)..."
	@uv run mypy app/
	@echo "Running Security Audit (bandit)..."
	@uv run bandit -r app/ -c pyproject.toml --severity-level medium --quiet
	@echo "All security checks passed."

# 5. Testing: Execution with coverage enforcement (80% threshold)
test: ## pytest with coverage (80% gate), HTML report in coverage_html/
	@echo "Running tests with coverage reporting..."
	@uv run pytest tests/ \
		--cov=app \
		--cov-report=term-missing \
		--cov-fail-under=80 \
		--cov-report=html:coverage_html

# 6. Cleanup: Remove temporary build and cache files
clean: ## Remove pycache/build artifacts and coverage output
	@echo "Removing pycache and build artifacts..."
	@find . -type d -name "__pycache__" -exec rm -rf {} +
	@find . -type f -name "*.pyc" -delete
	@rm -rf coverage_html/ .coverage
	@echo "Cleaned."

# 7. Purge: Destructive removal of project structure
purge: ## DESTRUCTIVE: remove app/tests/scripts/.github + top-level project files
	@echo "DESTRUCTIVE: Removing all project files!"
	@rm -rf app tests docker scripts .github pyproject.toml Dockerfile docker-compose.yml README.md

# 8. Train: Colab GPU if available, else local GPU (CUDA/MPS), else CPU
train: ## Train (Colab GPU if available, else local GPU, else CPU)
	@echo "Starting training job..."
	@uv run --group train python -m scripts.train_dispatch
	@echo "Training complete."

# 9. Evaluate: Run model validation on a test set
evaluate: ## Evaluate checkpoints/model_latest.pth against the test set
	@echo "Running model evaluation..."
	@uv run python -m app.ml.train.evaluate --checkpoint=checkpoints/model_latest.pth
	@echo "Evaluation metrics generated."

# 10. MLflow UI: Browse tracked runs (params, metrics, models)
# Port 5000 is avoided on purpose: macOS's AirPlay Receiver claims it by
# default and serves its own 403 Forbidden page instead of proxying through.
mlflow-ui: ## Browse MLflow experiment tracking UI at :5001
	@echo "Starting MLflow UI at http://127.0.0.1:5001 ..."
	@uv run --group train mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5001

# 11. Docker Build: Build the production image (multi-stage, see Dockerfile)
# GIT_SHA is baked in as a build arg so GET /version reports something
# useful even for a locally-built image, not just ones built by the CD
# pipeline (see Dockerfile's ARG GIT_SHA / APP_GIT_SHA).
docker-build: ## Build the production Docker image
	@echo "Building Docker image $(IMAGE_NAME):latest ..."
	@docker build \
		--build-arg GIT_SHA=$$(git rev-parse --short HEAD 2>/dev/null || echo unknown) \
		-t $(IMAGE_NAME):latest .

# 12. Docker Run: Serve the image at http://localhost:8000
# Mounts ./checkpoints so a locally trained model is used if present —
# the image itself ships with no checkpoint baked in (see Dockerfile).
docker-run: ## Run the image at :8000 (set WEB_CONCURRENCY=N to override)
	@echo "Starting $(CONTAINER_NAME) at http://localhost:8000 (WEB_CONCURRENCY=$(WEB_CONCURRENCY)) ..."
	@docker run --rm -d \
		--name $(CONTAINER_NAME) \
		-p 8000:8000 \
		-e WEB_CONCURRENCY=$(WEB_CONCURRENCY) \
		-v $(CURDIR)/checkpoints:/app/checkpoints \
		$(IMAGE_NAME):latest

# 13. Docker Stop: Stop and remove the running container
docker-stop: ## Stop and remove the running container
	@echo "Stopping $(CONTAINER_NAME) ..."
	@docker stop $(CONTAINER_NAME)

# 14. Docker Logs: Tail logs from the running container
docker-logs: ## Tail logs from the running container
	@docker logs -f $(CONTAINER_NAME)

# 15. Promote: Point model_latest.pth at a specific training run
promote: ## Point model_latest.pth at RUN_ID=<run-id> (see promote-list); FORCE=1 skips the promotion gate
	@uv run python -m app.ml.train.promote --run-id=$(RUN_ID) $(if $(FORCE),--force,)

promote-list: ## List MLflow runs available to promote (marks champion/shadow)
	@uv run python -m app.ml.train.promote --list

promote-shadow: ## Stage RUN_ID as the shadow model (scored on live traffic, never served) — see promote-commit
	@uv run python -m app.ml.train.promote --shadow=$(RUN_ID)

promote-commit: ## Promote the staged shadow to champion (same gate as promote); FORCE=1 skips it
	@uv run python -m app.ml.train.promote --commit $(if $(FORCE),--force,)

# 16. Export ONNX: For runtimes other than PyTorch (ONNX Runtime, TensorRT).
# Not part of the default dependency set — installs its own group on demand.
export-onnx: ## Export the trained model to ONNX format
	@uv run --group onnx python -m app.ml.export_onnx
