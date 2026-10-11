from unittest.mock import patch

import pytest

from sensai.mcp.config import MCPServerConfig
from sensai.mcp.errors import MCPStartupError
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

    async def initialize(self) -> None:
        self.initialized = True


@pytest.mark.anyio
async def test_starts_and_initializes_session() -> None:
    streams = ("read-stream", "write-stream")
    stdio = AsyncContext(streams)
    client = FakeClientSession()
    config = MCPServerConfig(command="fake-server", args=["--test"])
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
    config = MCPServerConfig(command="missing-server")
    session = MCPServerSession(config)
    stdio = AsyncContext((), error=RuntimeError("could not start"))

    with patch("sensai.mcp.session.stdio_client", return_value=stdio), pytest.raises(
        MCPStartupError,
        match="could not start",
    ):
        await session.start()


@pytest.mark.anyio
async def test_closes_session_and_server_process() -> None:
    stdio = AsyncContext(("read-stream", "write-stream"))
    client_context = AsyncContext(FakeClientSession())
    session = MCPServerSession(MCPServerConfig(command="fake-server"))

    with (
        patch("sensai.mcp.session.stdio_client", return_value=stdio),
        patch("sensai.mcp.session.ClientSession", return_value=client_context),
    ):
        await session.start()
        await session.close()

    assert client_context.closed is True
    assert stdio.closed is True
