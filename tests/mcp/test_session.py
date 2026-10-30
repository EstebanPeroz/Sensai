from unittest.mock import AsyncMock, patch

import pytest

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


@pytest.mark.anyio
async def test_calls_tool_sync_from_an_active_event_loop() -> None:
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
        result = session.call_tool_sync("echo", {"message": "hello"})

    assert result == "Echoed message"
    assert client.initialized is True
    assert client_context.closed is True


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
async def test_refuses_http_transport_until_it_is_supported() -> None:
    session = MCPServerSession(MCPServerConfig(name="remote", transport="http", url="http://localhost/mcp"))

    with (
        patch("sensai.mcp.session.stdio_client") as stdio_client,
        pytest.raises(MCPStartupError, match="transport 'http' is not supported yet"),
    ):
        await session.start()

    stdio_client.assert_not_called()
