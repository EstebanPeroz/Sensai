from pathlib import Path

import pytest

from sensai.main import main


class TestMain:
    def test_returns_zero_on_valid_settings(self) -> None:
        assert main([]) == 0

    def test_reports_config_error_without_traceback(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        assert main(["--config", str(tmp_path / "missing.toml")]) == 2

        err = capsys.readouterr().err
        assert err.startswith("sensai: error: config file not found")
        assert "Traceback" not in err
