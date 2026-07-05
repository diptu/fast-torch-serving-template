from pydantic_settings import SettingsConfigDict

from app.core.config import Settings


def test_yaml_file_overrides_field_defaults(tmp_path) -> None:
    yaml_path = tmp_path / "test.yaml"
    yaml_path.write_text("SEED: 999\n")

    class TestSettings(Settings):
        model_config = SettingsConfigDict(yaml_file=str(yaml_path))

    assert TestSettings().SEED == 999


def test_env_var_overrides_yaml_file(tmp_path, monkeypatch) -> None:
    yaml_path = tmp_path / "test.yaml"
    yaml_path.write_text("SEED: 999\n")
    monkeypatch.setenv("APP_SEED", "111")

    class TestSettings(Settings):
        model_config = SettingsConfigDict(yaml_file=str(yaml_path))

    assert TestSettings().SEED == 111


def test_missing_yaml_file_falls_back_to_field_defaults(tmp_path) -> None:
    class TestSettings(Settings):
        model_config = SettingsConfigDict(yaml_file=str(tmp_path / "nonexistent.yaml"))

    assert TestSettings().SEED == 42
