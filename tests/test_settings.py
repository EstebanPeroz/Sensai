from pathlib import Path

import pytest

from sensai.config.settings import ConfigError, OllamaSettings, Settings, load_settings


def write_config(tmp_path: Path, content: str) -> Path:
    path = tmp_path / "settings.toml"
    path.write_text(content)
    return path


class TestPackagedSettings:
    def test_loads_packaged_file_by_default(self) -> None:
        settings = load_settings()

        assert isinstance(settings, Settings)
        assert settings.llm.model != ""
        assert settings.ollama.base_url != ""


class TestPrecedence:
    def test_missing_keys_fall_back_to_defaults(self, tmp_path: Path) -> None:
        settings = load_settings(write_config(tmp_path, '[llm]\nmodel = "llama3.2"\n'))

        assert settings.llm.model == "llama3.2"
        assert settings.ollama == OllamaSettings()

    def test_empty_file_gives_defaults(self, tmp_path: Path) -> None:
        assert load_settings(write_config(tmp_path, "")) == Settings()


class TestValidation:
    @pytest.mark.parametrize(
        "content",
        [
            '[llm]\nmodle = "x"\n',
            'llm = "x"\n',
        ],
    )
    def test_invalid_settings_raise_config_error(self, tmp_path: Path, content: str) -> None:
        with pytest.raises(ConfigError, match="invalid settings"):
            load_settings(write_config(tmp_path, content))


class TestFileErrors:
    def test_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(ConfigError, match="config file not found"):
            load_settings(tmp_path / "nope.toml")

    def test_directory(self, tmp_path: Path) -> None:
        with pytest.raises(ConfigError, match="config path is a directory"):
            load_settings(tmp_path)

    def test_invalid_toml(self, tmp_path: Path) -> None:
        with pytest.raises(ConfigError, match="invalid TOML"):
            load_settings(write_config(tmp_path, "[llm\nmodel = \n"))
