import os
import signal
import socket
import sys
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, patch

import anyio
import pytest
from sensai_server.config.settings import MCPServerConfig
from sensai_server.tools.mcp.errors import MCPCallError, MCPProtocolError, MCPStartupError
from sensai_server.tools.mcp.session import MCPServerSession

FIXTURE_SERVER = Path(__file__).parent / "fixtures" / "tiny_server.py"
SESSION = "sensai_server.tools.mcp.session"

pytestmark = pytest.mark.anyio


class AsyncContext:
    def __init__(self, value: object, *, error: Exception | None = None) -> None:
        self.value = value
        self.error = error
        self.closed = False

    async def __aenter__(self) -> object:
        if self.error is not None:
            raise self.error
        return self.value

    async def __aexit__(self, *_args: object) -> None:
        self.closed = True


class FakeClientSession:
    def __init__(self) -> None:
        self.initialized = False
        self.tools: list[object] = []
        self.call_result: object = None
        self.call_error: Exception | None = None

    async def initialize(self) -> None:
        self.initialized = True

    async def list_tools(self) -> object:
        return type("ToolList", (), {"tools": self.tools})()

    async def call_tool(self, _name: str, _arguments: dict[str, object]) -> object:
        if self.call_error is not None:
            raise self.call_error
        return self.call_result


def text_result(text: str) -> object:
    return type("CallResult", (), {"isError": False, "content": [type("Text", (), {"type": "text", "text": text})()]})()


def fake_transport(client: FakeClientSession | AsyncContext) -> tuple[Any, Any]:
    """Patch the stdio transport and the client session with fakes, the client session yielding `client`."""
    context = client if isinstance(client, AsyncContext) else AsyncContext(client)
    return (
        patch(f"{SESSION}.stdio_client", return_value=AsyncContext(("read", "write"))),
        patch(f"{SESSION}.ClientSession", return_value=context),
    )


def fixture_config(**overrides: object) -> MCPServerConfig:
    return MCPServerConfig(name="tiny", command=sys.executable, args=[str(FIXTURE_SERVER)], **overrides)  # type: ignore[arg-type]


class TestStart:
    async def test_starts_and_initializes_session(self) -> None:
        streams = ("read-stream", "write-stream")
        client = FakeClientSession()
        session = MCPServerSession(MCPServerConfig(name="server", command="fake-server", args=["--test"]))

        with (
            patch(f"{SESSION}.stdio_client", return_value=AsyncContext(streams)) as stdio_client,
            patch(f"{SESSION}.ClientSession", return_value=AsyncContext(client)) as client_session,
        ):
            await session.start()

        stdio_client.assert_called_once()
        client_session.assert_called_once_with(*streams)
        assert client.initialized is True

    async def test_raises_startup_error_when_server_fails_to_start(self) -> None:
        session = MCPServerSession(MCPServerConfig(name="server", command="missing-server"))
        stdio = AsyncContext((), error=RuntimeError("could not start"))

        with (
            patch(f"{SESSION}.stdio_client", return_value=stdio),
            pytest.raises(MCPStartupError, match="could not start"),
        ):
            await session.start()

    async def test_closes_server_process_when_session_initialization_fails(self) -> None:
        stdio = AsyncContext(("read-stream", "write-stream"))
        client = FakeClientSession()
        client.initialize = AsyncMock(side_effect=RuntimeError("initialization failed"))  # type: ignore[method-assign]
        client_context = AsyncContext(client)
        session = MCPServerSession(MCPServerConfig(name="server", command="fake-server"))

        with (
            patch(f"{SESSION}.stdio_client", return_value=stdio),
            patch(f"{SESSION}.ClientSession", return_value=client_context),
            pytest.raises(MCPStartupError, match="initialization failed"),
        ):
            await session.start()

        assert client_context.closed is True
        assert stdio.closed is True

    async def test_closes_session_and_server_process(self) -> None:
        stdio = AsyncContext(("read-stream", "write-stream"))
        client_context = AsyncContext(FakeClientSession())
        session = MCPServerSession(MCPServerConfig(name="server", command="fake-server"))

        with (
            patch(f"{SESSION}.stdio_client", return_value=stdio),
            patch(f"{SESSION}.ClientSession", return_value=client_context),
        ):
            await session.start()
            await session.close()

        assert client_context.closed is True
        assert stdio.closed is True

    async def test_rejects_tool_listing_before_startup(self) -> None:
        session = MCPServerSession(MCPServerConfig(name="server", command="fake-server"))

        with pytest.raises(MCPProtocolError, match="session is not started"):
            await session.list_tools()


class TestHttp:
    async def test_connects_over_streamable_http_with_the_configured_headers(self) -> None:
        streams = ("read-stream", "write-stream")
        client = FakeClientSession()
        config = MCPServerConfig(
            name="remote",
            transport="http",
            url="https://example.com/mcp",
            headers={"Authorization": "Bearer token"},
        )

        with (
            patch(f"{SESSION}.httpx2.AsyncClient", return_value=AsyncContext("http-client")) as async_client,
            patch(f"{SESSION}.streamable_http_client", return_value=AsyncContext(streams)) as http_transport,
            patch(f"{SESSION}.stdio_client") as stdio_client,
            patch(f"{SESSION}.ClientSession", return_value=AsyncContext(client)) as client_session,
        ):
            await MCPServerSession(config).start()

        assert async_client.call_args.kwargs["headers"] == {"Authorization": "Bearer token"}
        http_transport.assert_called_once_with("https://example.com/mcp", http_client="http-client")
        stdio_client.assert_not_called()
        client_session.assert_called_once_with(*streams)
        assert client.initialized is True

    async def test_closes_http_client_when_session_closes(self) -> None:
        http_client = AsyncContext("http-client")
        session = MCPServerSession(MCPServerConfig(name="remote", transport="http", url="http://localhost/mcp"))

        with (
            patch(f"{SESSION}.httpx2.AsyncClient", return_value=http_client),
            patch(f"{SESSION}.streamable_http_client", return_value=AsyncContext(("read", "write"))),
            patch(f"{SESSION}.ClientSession", return_value=AsyncContext(FakeClientSession())),
        ):
            await session.start()
            await session.close()

        assert http_client.closed is True

    async def test_reports_a_refused_http_connection_as_a_startup_error(self) -> None:
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            port = probe.getsockname()[1]
        session = MCPServerSession(
            MCPServerConfig(name="remote", transport="http", url=f"http://127.0.0.1:{port}/mcp"),
        )

        with anyio.fail_after(10), pytest.raises(MCPStartupError, match="connection attempts failed"):
            await session.start()


class TestCallTool:
    async def test_returns_one_text_result(self) -> None:
        client = FakeClientSession()
        client.call_result = text_result("Echoed message")
        session = MCPServerSession(MCPServerConfig(name="server", command="fake-server"))
        stdio, client_session = fake_transport(client)

        with stdio, client_session:
            await session.start()
            result = await session.call_tool("echo", {"message": "hello"})

        assert result == "Echoed message"

    async def test_converts_a_call_timeout_to_a_readable_mcp_call_error(self) -> None:
        client = FakeClientSession()

        async def never_answer(_name: str, _arguments: dict[str, object]) -> object:
            await anyio.sleep_forever()

        client.call_tool = never_answer  # type: ignore[method-assign]
        session = MCPServerSession(MCPServerConfig(name="server", command="fake-server", call_timeout=0.01))
        stdio, client_session = fake_transport(client)

        with stdio, client_session:
            await session.start()
            with pytest.raises(MCPCallError, match=r"no answer within 0\.01s"):
                await session.call_tool("echo", {})

    async def test_converts_invalid_arguments_to_mcp_call_error(self) -> None:
        client = FakeClientSession()
        client.call_error = ValueError("invalid arguments")
        session = MCPServerSession(MCPServerConfig(name="server", command="fake-server"))
        stdio, client_session = fake_transport(client)

        with stdio, client_session:
            await session.start()
            with pytest.raises(MCPCallError, match="invalid arguments"):
                await session.call_tool("echo", {"message": 123})

    async def test_converts_mcp_reported_error_to_mcp_call_error(self) -> None:
        client = FakeClientSession()
        client.call_result = type(
            "CallResult",
            (),
            {"isError": True, "content": [type("Text", (), {"type": "text", "text": "tool failed"})()]},
        )()
        session = MCPServerSession(MCPServerConfig(name="server", command="fake-server"))
        stdio, client_session = fake_transport(client)

        with stdio, client_session:
            await session.start()
            with pytest.raises(MCPCallError, match="MCP tool returned an error"):
                await session.call_tool("echo", {})

    async def test_rejects_a_call_before_startup(self) -> None:
        session = MCPServerSession(MCPServerConfig(name="server", command="fake-server"))

        with pytest.raises(MCPCallError, match="session is not started"):
            await session.call_tool("echo", {})


class TestServe:
    async def test_reports_tools_keeps_session_open_for_calls_then_closes_it(self) -> None:
        client = FakeClientSession()
        client.tools = [type("Tool", (), {"name": "echo", "description": "Echo", "input_schema": {"type": "object"}})()]
        client.call_result = text_result("Echoed message")
        client_context = AsyncContext(client)
        session = MCPServerSession(MCPServerConfig(name="server", command="fake-server"))
        stdio, client_session = fake_transport(client_context)

        with stdio, client_session:
            async with anyio.create_task_group() as task_group:
                stop = anyio.Event()
                tools = await task_group.start(session.serve, stop)
                first = await session.call_tool("echo", {"message": "hello"})
                second = await session.call_tool("echo", {"message": "again"})
                assert [tool.name for tool in tools] == ["echo"]
                assert first == second == "Echoed message"
                assert client_context.closed is False
                stop.set()

        assert client_context.closed is True
        with pytest.raises(MCPCallError, match="session is not started"):
            await session.call_tool("echo", {})

    async def test_closes_session_when_tool_listing_fails(self) -> None:
        client = FakeClientSession()
        client.list_tools = AsyncMock(side_effect=RuntimeError("listing failed"))  # type: ignore[method-assign]
        client_context = AsyncContext(client)
        session = MCPServerSession(MCPServerConfig(name="server", command="fake-server"))
        stdio, client_session = fake_transport(client_context)

        with stdio, client_session:
            async with anyio.create_task_group() as task_group:
                with pytest.raises(MCPStartupError, match="listing failed"):
                    await task_group.start(session.serve, anyio.Event())

        assert client_context.closed is True


class TestFixtureServer:
    async def test_starts_lists_and_calls(self) -> None:
        session = MCPServerSession(fixture_config())

        async with anyio.create_task_group() as task_group:
            stop = anyio.Event()
            tools = await task_group.start(session.serve, stop)
            result = await session.call_tool("echo", {"message": "hello"})
            stop.set()

            assert {"echo", "fail", "image", "slow", "pid"} <= {tool.name for tool in tools}
            echo = next(tool for tool in tools if tool.name == "echo")
            assert echo.description == "Echo a message"
            assert "message" in echo.input_schema["properties"]  # type: ignore[operator]
            assert result == "hello"

    async def test_reports_a_bad_command_as_a_startup_error(self) -> None:
        session = MCPServerSession(MCPServerConfig(name="missing", command="sensai-missing-mcp-server"))

        with anyio.fail_after(10), pytest.raises(MCPStartupError):
            await session.start()

    async def test_reports_a_silent_server_as_a_startup_timeout(self) -> None:
        silent = MCPServerConfig(
            name="silent",
            command=sys.executable,
            args=["-c", "import time; time.sleep(60)"],
            startup_timeout=0.5,
        )

        with anyio.fail_after(10), pytest.raises(MCPStartupError, match=r"no answer within 0\.5s"):
            await MCPServerSession(silent).start()

    async def test_refuses_an_error_result(self) -> None:
        session = MCPServerSession(fixture_config())
        await session.start()
        try:
            with pytest.raises(MCPCallError, match="MCP tool returned an error"):
                await session.call_tool("fail", {})
        finally:
            await session.close()

    async def test_refuses_a_non_text_result(self) -> None:
        session = MCPServerSession(fixture_config())
        await session.start()
        try:
            with pytest.raises(MCPCallError, match="MCP tool returned a non-text result"):
                await session.call_tool("image", {})
        finally:
            await session.close()

    async def test_times_out_a_slow_call(self) -> None:
        session = MCPServerSession(fixture_config(call_timeout=0.2))
        await session.start()
        try:
            with pytest.raises(MCPCallError, match=r"no answer within 0\.2s"):
                await session.call_tool("slow", {"seconds": 5})
        finally:
            await session.close()

    async def test_marks_the_session_lost_once_the_subprocess_is_killed(self) -> None:
        session = MCPServerSession(fixture_config(call_timeout=5))

        with anyio.fail_after(15):
            async with anyio.create_task_group() as task_group:
                await task_group.start(session.serve, anyio.Event())
                pid = int(await session.call_tool("pid", {}))
                os.kill(pid, signal.SIGKILL)
                # The transport sees the dead subprocess and cancels `serve`, which then returns on its own.
                with pytest.raises(MCPCallError, match="MCP server 'tiny' disconnected"):
                    await session.call_tool("echo", {"message": "hello"})

            with pytest.raises(MCPCallError, match="MCP server 'tiny' disconnected, restart the server to reconnect"):
                await session.call_tool("echo", {"message": "hello"})
