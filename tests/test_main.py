from pathlib import Path

import pytest

from sensai.main import build_settings, main, parse_args


class TestBuildSettings:
    def test_defaults_to_packaged_settings(self) -> None:
        settings = build_settings(parse_args([]))

        assert settings.llm.model == "qwen2.5:1.5b"

    def test_model_argument_overrides_config_file(self, tmp_path: Path) -> None:
        path = tmp_path / "settings.toml"
        path.write_text('[llm]\nmodel = "llama3.2"\n')

        settings = build_settings(parse_args(["--config", str(path), "--model", "mistral"]))

        assert settings.llm.model == "mistral"

    def test_config_argument_is_used(self, tmp_path: Path) -> None:
        path = tmp_path / "settings.toml"
        path.write_text('[llm]\nmodel = "llama3.2"\n')

        settings = build_settings(parse_args(["--config", str(path)]))

        assert settings.llm.model == "llama3.2"


class TestMain:
    def test_returns_zero_on_valid_settings(self) -> None:
        assert main([]) == 0

    def test_reports_config_error_without_traceback(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        assert main(["--config", str(tmp_path / "missing.toml")]) == 2

        err = capsys.readouterr().err
        assert err.startswith("sensai: error: config file not found")
        assert "Traceback" not in err
