# Kept in sync manually with pyproject.toml's [project] version — there's
# no [build-system] table here, so this project isn't reliably resolvable
# via importlib.metadata (in this venv or in the Docker image, which never
# ships pyproject.toml), and GET /version needs something to report.
__version__ = "0.1.0"
