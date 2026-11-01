import os
import signal
import socket
import subprocess
import sys
import threading
import time
from collections.abc import Iterator
from pathlib import Path
from unittest.mock import patch

import anyio
import pytest
from anyio.abc import TaskStatus
from sensai_server.config.settings import MCPServerConfig
from sensai_server.tools.mcp import MCPHost, MCPTool
from sensai_server.tools.mcp.discovery import MCPToolInfo
from sensai_server.tools.mcp.errors import MCPCallError, MCPStartupError

FIXTURE_SERVER = Path(__file__).parents[1] / "mcp" / "fixtures" / "tiny_server.py"
HOST = "sensai_server.tools.mcp.host"
SESSION = "sensai_server.tools.mcp.session"


def fixture_config(name: str = "local") -> MCPServerConfig:
    return MCPServerConfig(name=name, command=sys.executable, args=[str(FIXTURE_SERVER)])


def tool_named(tools: list[MCPTool], name: str) -> MCPTool:
    return next(tool for tool in tools if tool.name == name)


def call(host: MCPHost, tool: MCPTool, **arguments: object) -> str:
    """Call `tool` through the host from this thread, as `MCPTool.run` does from uvicorn's loop."""
    return host.call(tool._session, tool._info.name, arguments).result(timeout=10)  # noqa: SLF001


def is_running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


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
            assert call(host, tools[0], message="hello") == "hello"
        finally:
            host.close()

    def test_keeps_one_session_open_across_calls(self) -> None:
        import sensai_server.tools.mcp.session as session_module  # noqa: PLC0415

        host = MCPHost([fixture_config()])
        with patch(f"{SESSION}.stdio_client", wraps=session_module.stdio_client) as stdio_client:
            tools, _ = host.start()
            try:
                echo = tool_named(tools, "local__echo")
                results = [call(host, echo, message=str(index)) for index in range(3)]
            finally:
                host.close()

        assert results == ["0", "1", "2"]
        stdio_client.assert_called_once()

    def test_two_calls_hit_the_same_subprocess(self) -> None:
        host = MCPHost([fixture_config()])
        tools, _ = host.start()
        try:
            pid = tool_named(tools, "local__pid")
            first, second = call(host, pid), call(host, pid)
        finally:
            host.close()

        assert first == second

    def test_serves_calls_made_concurrently(self) -> None:
        host = MCPHost([fixture_config()])
        tools, _ = host.start()
        try:
            slow = tool_named(tools, "local__slow")
            started_at = time.perf_counter()
            futures = [host.call(slow._session, "slow", {"seconds": 0.5}) for _ in range(3)]  # noqa: SLF001
            results = [future.result(timeout=10) for future in futures]
            elapsed = time.perf_counter() - started_at
        finally:
            host.close()

        assert results == ["done"] * 3
        assert elapsed < 1.4

    def test_skips_a_missing_server_next_to_a_working_one(self) -> None:
        host = MCPHost([MCPServerConfig(name="missing", command="sensai-missing-mcp-server"), fixture_config()])
        tools, failures = host.start()
        try:
            assert [tool.name for tool in tools][:2] == ["local__echo", "local__reverse_echo"]
            assert list(failures) == ["missing"]
            assert isinstance(failures["missing"], MCPStartupError)
            assert call(host, tool_named(tools, "local__echo"), message="still here") == "still here"
        finally:
            host.close()

    def test_close_ends_every_subprocess_and_the_thread(self) -> None:
        threads_before = set(threading.enumerate())
        host = MCPHost([fixture_config("first"), fixture_config("second")])
        tools, _ = host.start()
        pids = [int(call(host, tool_named(tools, f"{name}__pid"))) for name in ("first", "second")]

        host.close()

        assert pids[0] != pids[1]
        assert not any(is_running(pid) for pid in pids)
        assert not [thread for thread in set(threading.enumerate()) - threads_before if thread.is_alive()]

    def test_close_is_idempotent(self) -> None:
        host = MCPHost([fixture_config()])
        host.start()

        host.close()
        host.close()

    def test_refuses_calls_once_closed(self) -> None:
        host = MCPHost([fixture_config()])
        tools, _ = host.start()
        host.close()

        with pytest.raises(MCPCallError, match="MCP host is closed"):
            call(host, tool_named(tools, "local__echo"), message="too late")

    def test_does_not_start_again_once_closed(self) -> None:
        host = MCPHost([fixture_config()])
        host.start()
        host.close()

        assert host.start() == ([], {})

    def test_cancelling_the_future_cancels_the_call(self) -> None:
        host = MCPHost([fixture_config()])
        tools, _ = host.start()
        try:
            slow = tool_named(tools, "local__slow")
            future = host.call(slow._session, "slow", {"seconds": 30})  # noqa: SLF001
            time.sleep(0.2)
            assert future.cancel()
            # The session still answers: the cancelled call did not take it down.
            assert call(host, tool_named(tools, "local__echo"), message="after") == "after"
        finally:
            started_at = time.perf_counter()
            host.close()

        # The slow call no longer holds the session open.
        assert time.perf_counter() - started_at < 5


class TestHttpServer:
    @pytest.fixture
    def http_server(self, tmp_path: Path) -> Iterator[tuple[str, subprocess.Popen[bytes]]]:
        """Serve the fixture over http, returning its url and its process."""
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            port = probe.getsockname()[1]
        with (tmp_path / "server-stderr.log").open("wb") as stderr_file:
            server = subprocess.Popen(  # noqa: S603
                [sys.executable, str(FIXTURE_SERVER), "--http", str(port)],
                stdout=subprocess.DEVNULL,
                stderr=stderr_file,
            )
        try:
            anyio.run(_wait_for_port, port)
            yield f"http://127.0.0.1:{port}/mcp", server
        finally:
            server.terminate()
            server.wait()

    def test_discovers_and_calls_the_fixture_tool_over_http(
        self,
        http_server: tuple[str, subprocess.Popen[bytes]],
    ) -> None:
        url, _ = http_server
        host = MCPHost([MCPServerConfig(name="remote", transport="http", url=url)])
        tools, _ = host.start()
        try:
            assert [tool.name for tool in tools][:2] == ["remote__echo", "remote__reverse_echo"]
            assert call(host, tools[0], message="hello over http") == "hello over http"
        finally:
            host.close()

    def test_a_server_going_down_leaves_the_others_open(
        self,
        http_server: tuple[str, subprocess.Popen[bytes]],
    ) -> None:
        url, server = http_server
        host = MCPHost([fixture_config(), MCPServerConfig(name="remote", transport="http", url=url)])
        tools, _ = host.start()
        try:
            remote_echo = tool_named(tools, "remote__echo")
            server.send_signal(signal.SIGTERM)
            server.wait()

            # The first call finds the connection gone, the next ones are refused right away.
            with pytest.raises(MCPCallError):
                call(host, remote_echo, message="gone")
            with pytest.raises(MCPCallError, match="MCP server 'remote' disconnected, restart the server"):
                call(host, remote_echo, message="gone")
            assert call(host, tool_named(tools, "local__echo"), message="still here") == "still here"
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

    async def serve(self, stop: anyio.Event, *, task_status: TaskStatus[list[MCPToolInfo]]) -> None:
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
    with patch(f"{HOST}.MCPServerSession", FakeSession):
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

    def test_reports_failures_in_configuration_order(
        self,
        fake_sessions: dict[str, tuple[float, Exception | None]],
    ) -> None:
        fake_sessions.update(
            {"late": (0.05, MCPStartupError("late")), "early": (0, MCPStartupError("early"))},
        )
        host = MCPHost([fixture_config("late"), fixture_config("early")])

        _, failures = host.start()
        host.close()

        assert list(failures) == ["late", "early"]

    @pytest.mark.usefixtures("fake_sessions")
    def test_skips_disabled_servers(self) -> None:
        disabled = MCPServerConfig(name="disabled", command="server", enabled=False)
        host = MCPHost([disabled, fixture_config("enabled")])

        tools, _ = host.start()
        host.close()

        assert FakeSession.started_names == ["enabled"]
        assert [tool.name for tool in tools] == ["enabled__echo"]

    def test_starts_no_thread_without_enabled_server(self) -> None:
        host = MCPHost([MCPServerConfig(name="disabled", command="server", enabled=False)])

        with patch(f"{HOST}.start_blocking_portal") as start_blocking_portal:
            assert host.start() == ([], {})
            host.close()

        start_blocking_portal.assert_not_called()

    def test_refuses_calls_before_start(self) -> None:
        host = MCPHost([fixture_config()])

        with pytest.raises(MCPCallError, match="MCP host is closed"):
            host.call(object(), "echo", {})  # type: ignore[arg-type]

    def test_close_before_start_does_nothing(self) -> None:
        MCPHost([fixture_config()]).close()
