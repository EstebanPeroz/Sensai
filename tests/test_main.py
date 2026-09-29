from pathlib import Path

import pytest

from sensai.main import build_settings, main, parse_args


def write_config(tmp_path: Path) -> Path:
    path = tmp_path / "settings.toml"
    path.write_text('[llm]\nmodel = "llama3.2"\n')
    return path


class TestBuildSettings:
    def test_defaults_to_packaged_settings(self) -> None:
        settings = build_settings(parse_args([]))

        assert settings.llm.model is None
        assert settings.llm.providers != []

    def test_model_argument_overrides_config_file(self, tmp_path: Path) -> None:
        settings = build_settings(parse_args(["--config", str(write_config(tmp_path)), "--model", "mistral"]))

        assert settings.llm.model == "mistral"

    def test_config_argument_is_used(self, tmp_path: Path) -> None:
        settings = build_settings(parse_args(["--config", str(write_config(tmp_path))]))

        assert settings.llm.model == "llama3.2"


class TestMain:
    def test_returns_zero_on_valid_settings(self) -> None:
        assert main([]) == 0

    def test_reports_config_error_without_traceback(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        assert main(["--config", str(tmp_path / "missing.toml")]) == 2

        err = capsys.readouterr().err
        assert err.startswith("sensai: error: config file not found")
        assert "Traceback" not in err
