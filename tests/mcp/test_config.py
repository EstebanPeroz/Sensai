import pytest

from sensai.mcp.config import MCPServerConfig


class TestMCPServerConfig:
    def test_uses_default_values(self) -> None:
        config = MCPServerConfig(name="server", command="uv")

        assert config.args == []
        assert config.enabled is True
        assert config.startup_timeout == 10.0
        assert config.call_timeout == 30.0

    def test_accepts_custom_values(self) -> None:
        config = MCPServerConfig(
            name="server",
            command="python",
            args=["server.py"],
            enabled=False,
            startup_timeout=5.0,
            call_timeout=15.0,
        )

        assert config.name == "server"
        assert config.command == "python"
        assert config.args == ["server.py"]
        assert config.enabled is False
        assert config.startup_timeout == 5.0
        assert config.call_timeout == 15.0

    @pytest.mark.parametrize("name", ["", "my server", "files.local", "srv/1"])
    def test_rejects_invalid_name(self, name: str) -> None:
        with pytest.raises(ValueError, match="server name must be non-empty"):
            MCPServerConfig(name=name, command="uv")

    @pytest.mark.parametrize("name", ["filesystem", "my-server", "server_2"])
    def test_accepts_valid_name(self, name: str) -> None:
        assert MCPServerConfig(name=name, command="uv").name == name

    def test_rejects_empty_command(self) -> None:
        with pytest.raises(ValueError, match="command must be a non-empty string"):
            MCPServerConfig(name="server", command="")

    @pytest.mark.parametrize("timeout", [0, -1])
    def test_rejects_non_positive_startup_timeout(self, timeout: float) -> None:
        with pytest.raises(ValueError, match="startup timeout must be greater than zero"):
            MCPServerConfig(name="server", command="uv", startup_timeout=timeout)

    @pytest.mark.parametrize("timeout", [0, -1])
    def test_rejects_non_positive_call_timeout(self, timeout: float) -> None:
        with pytest.raises(ValueError, match="call timeout must be greater than zero"):
            MCPServerConfig(name="server", command="uv", call_timeout=timeout)
