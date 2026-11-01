import signal
import socket
import subprocess
import sys
import time
from collections.abc import Iterator, Sequence
from pathlib import Path
from unittest.mock import patch

import anyio
import pytest
from anyio.abc import TaskStatus

import sensai.mcp.session
from sensai.mcp.client import MCPHost
from sensai.mcp.config import MCPServerConfig
from sensai.mcp.discovery import MCPToolInfo
from sensai.mcp.errors import MCPStartupError
from sensai.tools.tool import Tool
from sensai.tools.tool_manager import ToolManager

FIXTURE_SERVER = Path(__file__).parent / "fixtures" / "tiny_server.py"


def fixture_config(name: str = "local") -> MCPServerConfig:
    return MCPServerConfig(name=name, command=sys.executable, args=[str(FIXTURE_SERVER)])


def manager_of(tools: Sequence[Tool[str]]) -> ToolManager:
    return ToolManager(list(tools))


def echo(manager: ToolManager, name: str, message: str) -> str:
    return manager.call({"function": {"name": name, "arguments": {"message": message}}})["content"]


class TestRealServers:
    def test_discovers_and_calls_the_fixture_tool(self) -> None:
        host = MCPHost([fixture_config()])
        tools, failures = host.start()
        try:
            assert failures == {}
            assert [tool.name for tool in tools][:2] == ["local__echo", "local__reverse_echo"]
            assert tools[0].tool_rule == {
                "type": "function",
                "function": {
                    "name": "local__echo",
                    "description": "Echo a message",
                    "parameters": {
                        "properties": {"message": {"title": "Message", "type": "string"}},
                        "required": ["message"],
                        "title": "echoArguments",
                        "type": "object",
                    },
                },
            }
            assert echo(manager_of(tools), "local__echo", "hello") == "hello"
        finally:
            host.close()

    def test_keeps_one_session_open_across_calls(self) -> None:
        host = MCPHost([fixture_config()])
        with patch("sensai.mcp.session.stdio_client", wraps=sensai.mcp.session.stdio_client) as stdio_client:
            tools, _ = host.start()
            try:
                manager = manager_of(tools)
                results = [echo(manager, "local__echo", str(index)) for index in range(3)]
            finally:
                host.close()

        assert results == ["0", "1", "2"]
        stdio_client.assert_called_once()

    def test_skips_a_missing_server_next_to_a_working_one(self) -> None:
        host = MCPHost([MCPServerConfig(name="missing", command="sensai-missing-mcp-server"), fixture_config()])
        tools, failures = host.start()
        try:
            assert [tool.name for tool in tools][:2] == ["local__echo", "local__reverse_echo"]
            assert list(failures) == ["missing"]
            assert isinstance(failures["missing"], MCPStartupError)
            assert echo(manager_of(tools), "local__echo", "still here") == "still here"
        finally:
            host.close()

    def test_close_ends_the_sessions(self) -> None:
        host = MCPHost([fixture_config()])
        tools, _ = host.start()
        host.close()

        assert echo(manager_of(tools), "local__echo", "too late") == "error: MCP session is not started"


class TestHttpServer:
    @pytest.fixture
    def http_server(self, tmp_path: Path) -> Iterator[tuple[str, subprocess.Popen[bytes], Path]]:
        """Serve the fixture over http, returning its url, its process, and the file receiving its stderr."""
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            port = probe.getsockname()[1]
        stderr = tmp_path / "server-stderr.log"
        with stderr.open("wb") as stderr_file:
            server = subprocess.Popen(  # noqa: S603
                [sys.executable, str(FIXTURE_SERVER), "--http", str(port)],
                stdout=subprocess.DEVNULL,
                stderr=stderr_file,
            )
        try:
            anyio.run(_wait_for_port, port)
            yield f"http://127.0.0.1:{port}/mcp", server, stderr
        finally:
            server.terminate()
            server.wait()

    def test_discovers_and_calls_the_fixture_tool_over_http(
        self,
        http_server: tuple[str, subprocess.Popen[bytes], Path],
    ) -> None:
        url, _, _ = http_server
        host = MCPHost([MCPServerConfig(name="remote", transport="http", url=url)])
        tools, _ = host.start()
        try:
            assert [tool.name for tool in tools][:2] == ["remote__echo", "remote__reverse_echo"]
            assert echo(manager_of(tools), "remote__echo", "hello over http") == "hello over http"
        finally:
            host.close()

    def test_reverse_echo_shows_the_message_on_the_server_side(
        self,
        http_server: tuple[str, subprocess.Popen[bytes], Path],
    ) -> None:
        url, _, stderr = http_server
        host = MCPHost([MCPServerConfig(name="remote", transport="http", url=url)])
        tools, _ = host.start()
        try:
            result = echo(manager_of(tools), "remote__reverse_echo", "shown over there")
        finally:
            host.close()

        assert result == "The message was shown in the terminal of the server."
        assert "[reverse_echo] shown over there" in stderr.read_text()

    def test_a_server_going_down_leaves_the_others_open(
        self,
        http_server: tuple[str, subprocess.Popen[bytes], Path],
    ) -> None:
        url, server, _ = http_server
        host = MCPHost([fixture_config(), MCPServerConfig(name="remote", transport="http", url=url)])
        tools, _ = host.start()
        try:
            manager = manager_of(tools)
            server.send_signal(signal.SIGTERM)
            server.wait()

            # The first call finds the connection gone, the next ones are refused right away.
            assert echo(manager, "remote__echo", "gone").startswith("error: ")
            assert echo(manager, "remote__echo", "gone").startswith(
                "error: MCP server 'remote' disconnected, restart Sensai to reconnect: ",
            )
            assert echo(manager, "local__echo", "still here") == "still here"
        finally:
            host.close()


async def _wait_for_port(port: int) -> None:
    with anyio.fail_after(10):
        while True:
            try:
                stream = await anyio.connect_tcp("127.0.0.1", port)
            except OSError:
                await anyio.sleep(0.05)
            else:
                await stream.aclose()
                return


class FakeSession:
    """Session whose `serve` reports one tool after a delay, or fails, then stays open until `stop` is set."""

    behaviours: dict[str, tuple[float, Exception | None]]
    started_names: list[str]

    def __init__(self, config: MCPServerConfig) -> None:
        self.server_name = config.name

    async def serve(
        self,
        _: object,
        stop: anyio.Event,
        *,
        task_status: TaskStatus[list[MCPToolInfo]],
    ) -> None:
        delay, error = self.behaviours.get(self.server_name, (0, None))
        self.started_names.append(self.server_name)
        await anyio.sleep(delay)
        if error is not None:
            raise error
        task_status.started([MCPToolInfo("echo", None, {"type": "object"})])
        await stop.wait()


@pytest.fixture
def fake_sessions() -> Iterator[dict[str, tuple[float, Exception | None]]]:
    FakeSession.behaviours = {}
    FakeSession.started_names = []
    with patch("sensai.mcp.client.MCPServerSession", FakeSession):
        yield FakeSession.behaviours


class TestStartup:
    def test_returns_tools_in_configuration_order(self, fake_sessions: dict[str, tuple[float, None]]) -> None:
        fake_sessions["slow"] = (0.05, None)
        host = MCPHost([fixture_config("slow"), fixture_config("fast")])

        tools, failures = host.start()
        host.close()

        assert [tool.name for tool in tools] == ["slow__echo", "fast__echo"]
        assert failures == {}

    def test_starts_servers_concurrently(self, fake_sessions: dict[str, tuple[float, None]]) -> None:
        fake_sessions.update({"first": (0.5, None), "second": (0.5, None)})
        host = MCPHost([fixture_config("first"), fixture_config("second")])

        started_at = time.perf_counter()
        host.start()
        elapsed = time.perf_counter() - started_at
        host.close()

        assert elapsed < 0.9

    def test_reports_failed_servers_by_name(self, fake_sessions: dict[str, tuple[float, Exception | None]]) -> None:
        error = MCPStartupError("server failed")
        fake_sessions["broken"] = (0, error)
        host = MCPHost([fixture_config("broken"), fixture_config("working")])

        tools, failures = host.start()
        host.close()

        assert [tool.name for tool in tools] == ["working__echo"]
        assert failures == {"broken": error}

    @pytest.mark.usefixtures("fake_sessions")
    def test_skips_disabled_servers(self) -> None:
        disabled = MCPServerConfig(name="disabled", command="server", enabled=False)
        host = MCPHost([disabled, fixture_config("enabled")])

        tools, _ = host.start()
        host.close()

        assert FakeSession.started_names == ["enabled"]
        assert [tool.name for tool in tools] == ["enabled__echo"]

    def test_starts_no_event_loop_without_enabled_server(self) -> None:
        host = MCPHost([MCPServerConfig(name="disabled", command="server", enabled=False)])

        with patch("sensai.mcp.client.start_blocking_portal") as start_blocking_portal:
            assert host.start() == ([], {})
            host.close()

        start_blocking_portal.assert_not_called()

    def test_close_before_start_does_nothing(self) -> None:
        MCPHost([fixture_config()]).close()
