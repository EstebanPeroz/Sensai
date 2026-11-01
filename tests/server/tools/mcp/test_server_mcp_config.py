import pytest
from sensai_server.config.settings import MCPServerConfig


class TestMCPServerConfig:
    def test_uses_default_values(self) -> None:
        config = MCPServerConfig(name="server", command="uv")

        assert config.transport == "stdio"
        assert config.args == []
        assert config.url is None
        assert config.headers == {}
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


class TestHTTPTransport:
    def test_accepts_url_and_headers(self) -> None:
        config = MCPServerConfig(
            name="remote",
            transport="http",
            url="https://example.com/mcp",
            headers={"Authorization": "Bearer token"},
        )

        assert config.command is None
        assert config.url == "https://example.com/mcp"
        assert config.headers == {"Authorization": "Bearer token"}

    @pytest.mark.parametrize("url", [None, "", "example.com/mcp", "ftp://example.com/mcp", "https:///mcp"])
    def test_rejects_missing_or_invalid_url(self, url: str | None) -> None:
        with pytest.raises(ValueError, match=r"url must be an http\(s\) URL with a host"):
            MCPServerConfig(name="remote", transport="http", url=url)

    @pytest.mark.parametrize(("command", "args"), [("uv", []), (None, ["server.py"])])
    def test_rejects_command_and_args(self, command: str | None, args: list[str]) -> None:
        with pytest.raises(ValueError, match="command and args only apply to the stdio transport"):
            MCPServerConfig(
                name="remote", transport="http", url="http://localhost:8000/mcp", command=command, args=args
            )

    def test_rejects_non_string_header_values(self) -> None:
        with pytest.raises(ValueError, match="headers must map strings to strings"):
            MCPServerConfig(name="remote", transport="http", url="http://localhost/mcp", headers={"X-Retries": 3})  # type: ignore[dict-item]


class TestStdioTransport:
    def test_rejects_missing_command(self) -> None:
        with pytest.raises(ValueError, match="command must be a non-empty string"):
            MCPServerConfig(name="server")

    @pytest.mark.parametrize(
        ("url", "headers"),
        [("http://localhost/mcp", {}), (None, {"Authorization": "Bearer token"})],
    )
    def test_rejects_url_and_headers(self, url: str | None, headers: dict[str, str]) -> None:
        with pytest.raises(ValueError, match="url and headers only apply to the http transport"):
            MCPServerConfig(name="server", command="uv", url=url, headers=headers)


def test_rejects_unknown_transport() -> None:
    with pytest.raises(ValueError, match="transport must be 'stdio' or 'http', got 'sse'"):
        MCPServerConfig(name="server", transport="sse", command="uv")  # type: ignore[arg-type]
