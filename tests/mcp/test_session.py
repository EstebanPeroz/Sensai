import socket
from unittest.mock import AsyncMock, patch

import anyio
import pytest
from anyio.from_thread import BlockingPortal

from sensai.mcp.config import MCPServerConfig
from sensai.mcp.errors import MCPCallError, MCPProtocolError, MCPStartupError
from sensai.mcp.session import MCPServerSession


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


@pytest.mark.anyio
async def test_starts_and_initializes_session() -> None:
    streams = ("read-stream", "write-stream")
    stdio = AsyncContext(streams)
    client = FakeClientSession()
    config = MCPServerConfig(name="server", command="fake-server", args=["--test"])
    session = MCPServerSession(config)

    with (
        patch("sensai.mcp.session.stdio_client", return_value=stdio) as stdio_client,
        patch("sensai.mcp.session.ClientSession", return_value=AsyncContext(client)) as client_session,
    ):
        await session.start()

    stdio_client.assert_called_once()
    client_session.assert_called_once_with(*streams)
    assert client.initialized is True


@pytest.mark.anyio
async def test_raises_startup_error_when_server_fails_to_start() -> None:
    config = MCPServerConfig(name="server", command="missing-server")
    session = MCPServerSession(config)
    stdio = AsyncContext((), error=RuntimeError("could not start"))

    with (
        patch("sensai.mcp.session.stdio_client", return_value=stdio),
        pytest.raises(
            MCPStartupError,
            match="could not start",
        ),
    ):
        await session.start()


@pytest.mark.anyio
async def test_closes_server_process_when_session_initialization_fails() -> None:
    stdio = AsyncContext(("read-stream", "write-stream"))
    client_context = AsyncContext(FakeClientSession())
    client_context.value.initialize = AsyncMock(side_effect=RuntimeError("initialization failed"))
    session = MCPServerSession(MCPServerConfig(name="server", command="fake-server"))

    with (
        patch("sensai.mcp.session.stdio_client", return_value=stdio),
        patch("sensai.mcp.session.ClientSession", return_value=client_context),
        pytest.raises(MCPStartupError, match="initialization failed"),
    ):
        await session.start()

    assert client_context.closed is True
    assert stdio.closed is True


@pytest.mark.anyio
async def test_closes_session_and_server_process() -> None:
    stdio = AsyncContext(("read-stream", "write-stream"))
    client_context = AsyncContext(FakeClientSession())
    session = MCPServerSession(MCPServerConfig(name="server", command="fake-server"))

    with (
        patch("sensai.mcp.session.stdio_client", return_value=stdio),
        patch("sensai.mcp.session.ClientSession", return_value=client_context),
    ):
        await session.start()
        await session.close()

    assert client_context.closed is True
    assert stdio.closed is True


@pytest.mark.anyio
async def test_rejects_tool_listing_before_startup() -> None:
    session = MCPServerSession(MCPServerConfig(name="server", command="fake-server"))

    with pytest.raises(MCPProtocolError, match="session is not started"):
        await session.list_tools()


@pytest.mark.anyio
async def test_calls_tool_and_returns_one_text_result() -> None:
    client = FakeClientSession()
    client.call_result = type(
        "CallResult",
        (),
        {"isError": False, "content": [type("Text", (), {"type": "text", "text": "Echoed message"})()]},
    )()
    session = MCPServerSession(MCPServerConfig(name="server", command="fake-server"))
    client_context = AsyncContext(client)

    with (
        patch("sensai.mcp.session.stdio_client", return_value=AsyncContext(("read", "write"))),
        patch("sensai.mcp.session.ClientSession", return_value=client_context),
    ):
        await session.start()
        result = await session.call_tool("echo", {"message": "hello"})

    assert result == "Echoed message"


def text_result(text: str) -> object:
    return type("CallResult", (), {"isError": False, "content": [type("Text", (), {"type": "text", "text": text})()]})()


@pytest.mark.anyio
async def test_serve_reports_tools_keeps_session_open_for_sync_calls_then_closes_it() -> None:
    client = FakeClientSession()
    client.tools = [type("Tool", (), {"name": "echo", "description": "Echo", "input_schema": {"type": "object"}})()]
    client.call_result = text_result("Echoed message")
    client_context = AsyncContext(client)
    session = MCPServerSession(MCPServerConfig(name="server", command="fake-server"))

    with (
        patch("sensai.mcp.session.stdio_client", return_value=AsyncContext(("read", "write"))),
        patch("sensai.mcp.session.ClientSession", return_value=client_context),
    ):
        async with BlockingPortal() as portal, anyio.create_task_group() as task_group:
            stop = anyio.Event()
            tools = await task_group.start(session.serve, portal, stop)
            first = await anyio.to_thread.run_sync(session.call_tool_sync, "echo", {"message": "hello"})
            second = await anyio.to_thread.run_sync(session.call_tool_sync, "echo", {"message": "again"})
            assert client_context.closed is False
            stop.set()

    assert [tool.name for tool in tools] == ["echo"]
    assert first == second == "Echoed message"
    assert client_context.closed is True
    with pytest.raises(MCPCallError, match="session is not started"):
        session.call_tool_sync("echo", {})


def test_rejects_sync_calls_before_serving() -> None:
    session = MCPServerSession(MCPServerConfig(name="server", command="fake-server"))

    with pytest.raises(MCPCallError, match="session is not started"):
        session.call_tool_sync("echo", {})


@pytest.mark.anyio
async def test_serve_closes_session_when_tool_listing_fails() -> None:
    client = FakeClientSession()
    client.list_tools = AsyncMock(side_effect=RuntimeError("listing failed"))  # type: ignore[method-assign]
    client_context = AsyncContext(client)
    session = MCPServerSession(MCPServerConfig(name="server", command="fake-server"))

    with (
        patch("sensai.mcp.session.stdio_client", return_value=AsyncContext(("read", "write"))),
        patch("sensai.mcp.session.ClientSession", return_value=client_context),
    ):
        async with BlockingPortal() as portal, anyio.create_task_group() as task_group:
            with pytest.raises(MCPStartupError, match="listing failed"):
                await task_group.start(session.serve, portal, anyio.Event())

    assert client_context.closed is True


@pytest.mark.anyio
async def test_converts_a_call_timeout_to_a_readable_mcp_call_error() -> None:
    client = FakeClientSession()

    async def never_answer(_name: str, _arguments: dict[str, object]) -> object:
        await anyio.sleep_forever()

    client.call_tool = never_answer  # type: ignore[method-assign]
    session = MCPServerSession(MCPServerConfig(name="server", command="fake-server", call_timeout=0.01))

    with (
        patch("sensai.mcp.session.stdio_client", return_value=AsyncContext(("read", "write"))),
        patch("sensai.mcp.session.ClientSession", return_value=AsyncContext(client)),
    ):
        await session.start()
        with pytest.raises(MCPCallError, match=r"no answer within 0\.01s"):
            await session.call_tool("echo", {})


@pytest.mark.anyio
async def test_converts_invalid_arguments_to_mcp_call_error() -> None:
    client = FakeClientSession()
    client.call_error = ValueError("invalid arguments")
    session = MCPServerSession(MCPServerConfig(name="server", command="fake-server"))
    client_context = AsyncContext(client)

    with (
        patch("sensai.mcp.session.stdio_client", return_value=AsyncContext(("read", "write"))),
        patch("sensai.mcp.session.ClientSession", return_value=client_context),
    ):
        await session.start()
        with pytest.raises(MCPCallError, match="invalid arguments"):
            await session.call_tool("echo", {"message": 123})


@pytest.mark.anyio
async def test_converts_mcp_reported_error_to_mcp_call_error() -> None:
    client = FakeClientSession()
    client.call_result = type(
        "CallResult",
        (),
        {"isError": True, "content": [type("Text", (), {"type": "text", "text": "tool failed"})()]},
    )()
    session = MCPServerSession(MCPServerConfig(name="server", command="fake-server"))
    client_context = AsyncContext(client)

    with (
        patch("sensai.mcp.session.stdio_client", return_value=AsyncContext(("read", "write"))),
        patch("sensai.mcp.session.ClientSession", return_value=client_context),
    ):
        await session.start()
        with pytest.raises(MCPCallError, match="MCP tool returned an error"):
            await session.call_tool("echo", {})


@pytest.mark.anyio
async def test_converts_disconnected_server_to_mcp_call_error() -> None:
    session = MCPServerSession(MCPServerConfig(name="server", command="fake-server"))

    with pytest.raises(MCPCallError, match="session is not started"):
        await session.call_tool("echo", {})


@pytest.mark.anyio
async def test_connects_over_streamable_http_with_the_configured_headers() -> None:
    streams = ("read-stream", "write-stream")
    client = FakeClientSession()
    config = MCPServerConfig(
        name="remote",
        transport="http",
        url="https://example.com/mcp",
        headers={"Authorization": "Bearer token"},
    )
    http_client = AsyncContext("http-client")

    with (
        patch("sensai.mcp.session.httpx2.AsyncClient", return_value=http_client) as async_client,
        patch("sensai.mcp.session.streamable_http_client", return_value=AsyncContext(streams)) as http_transport,
        patch("sensai.mcp.session.stdio_client") as stdio_client,
        patch("sensai.mcp.session.ClientSession", return_value=AsyncContext(client)) as client_session,
    ):
        await MCPServerSession(config).start()

    assert async_client.call_args.kwargs["headers"] == {"Authorization": "Bearer token"}
    http_transport.assert_called_once_with("https://example.com/mcp", http_client="http-client")
    stdio_client.assert_not_called()
    client_session.assert_called_once_with(*streams)
    assert client.initialized is True


@pytest.mark.anyio
async def test_closes_http_client_when_session_closes() -> None:
    http_client = AsyncContext("http-client")
    session = MCPServerSession(MCPServerConfig(name="remote", transport="http", url="http://localhost/mcp"))

    with (
        patch("sensai.mcp.session.httpx2.AsyncClient", return_value=http_client),
        patch("sensai.mcp.session.streamable_http_client", return_value=AsyncContext(("read", "write"))),
        patch("sensai.mcp.session.ClientSession", return_value=AsyncContext(FakeClientSession())),
    ):
        await session.start()
        await session.close()

    assert http_client.closed is True


@pytest.mark.anyio
async def test_reports_a_refused_http_connection_as_a_startup_error() -> None:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    session = MCPServerSession(MCPServerConfig(name="remote", transport="http", url=f"http://127.0.0.1:{port}/mcp"))

    with anyio.fail_after(10), pytest.raises(MCPStartupError, match="connection attempts failed"):
        await session.start()
