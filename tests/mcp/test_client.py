import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import anyio
import pytest

from sensai.mcp.client import MCPClient, start_clients
from sensai.mcp.config import MCPServerConfig
from sensai.mcp.discovery import MCPToolInfo
from sensai.mcp.errors import MCPStartupError

FIXTURE_SERVER = Path(__file__).parent / "fixtures" / "tiny_server.py"


@pytest.mark.anyio
async def test_starts_server_discovers_tools_and_creates_mcp_tools() -> None:
    session = AsyncMock()
    session.list_tools.return_value = [MCPToolInfo("echo", "Echo text", {"type": "object"})]
    config = MCPServerConfig(name="server", command="server")
    client = MCPClient(config)

    with patch("sensai.mcp.client.MCPServerSession", return_value=session) as session_type:
        tools = await client.start()

    session_type.assert_called_once_with(config)
    session.start.assert_awaited_once()
    session.list_tools.assert_awaited_once()
    assert len(tools) == 1
    assert tools[0].name == "echo"


@pytest.mark.anyio
async def test_disabled_server_returns_no_tools_without_starting() -> None:
    client = MCPClient(MCPServerConfig(name="server", command="server", enabled=False))

    with patch("sensai.mcp.client.MCPServerSession") as session_type:
        assert await client.start() == []

    session_type.assert_not_called()


@pytest.mark.anyio
async def test_start_closes_discovery_session() -> None:
    session = AsyncMock()
    session.list_tools.return_value = []
    client = MCPClient(MCPServerConfig(name="server", command="server"))

    with patch("sensai.mcp.client.MCPServerSession", return_value=session):
        await client.start()

    session.close.assert_awaited_once()


@pytest.mark.anyio
async def test_start_closes_server_when_tool_discovery_fails() -> None:
    session = AsyncMock()
    session.list_tools.side_effect = RuntimeError("discovery failed")
    client = MCPClient(MCPServerConfig(name="server", command="server"))

    with (
        patch("sensai.mcp.client.MCPServerSession", return_value=session),
        pytest.raises(
            RuntimeError,
            match="discovery failed",
        ),
    ):
        await client.start()

    session.close.assert_awaited_once()


@pytest.mark.anyio
async def test_starts_fixture_discovers_tool_and_creates_adapter() -> None:
    client = MCPClient(
        MCPServerConfig(
            name="server",
            command=sys.executable,
            args=[str(FIXTURE_SERVER)],
        ),
    )

    tools = await client.start()

    assert len(tools) == 1
    assert tools[0].name == "echo"
    assert tools[0].tool_rule == {
        "type": "function",
        "function": {
            "name": "echo",
            "description": "Echo a message",
            "parameters": {
                "properties": {"message": {"title": "Message", "type": "string"}},
                "required": ["message"],
                "title": "echoArguments",
                "type": "object",
            },
        },
    }


def make_client(start: AsyncMock) -> MagicMock:
    client = MagicMock(spec=MCPClient)
    client.start = start
    return client


@pytest.mark.anyio
async def test_start_clients_returns_tools_in_client_order() -> None:
    first_tool, second_tool = MagicMock(), MagicMock()

    async def slow_start() -> list[MagicMock]:
        await anyio.sleep(0.01)
        return [first_tool]

    clients = [make_client(AsyncMock(side_effect=slow_start)), make_client(AsyncMock(return_value=[second_tool]))]

    assert await start_clients(clients) == [first_tool, second_tool]


@pytest.mark.anyio
async def test_start_clients_starts_clients_concurrently() -> None:
    both_started = anyio.Event()
    started = 0

    async def start() -> list[MagicMock]:
        nonlocal started
        started += 1
        if started == 2:
            both_started.set()
        await both_started.wait()
        return []

    clients = [make_client(AsyncMock(side_effect=start)), make_client(AsyncMock(side_effect=start))]

    with anyio.fail_after(1):
        assert await start_clients(clients) == []


@pytest.mark.anyio
async def test_start_clients_raises_the_failure_itself_instead_of_a_group() -> None:
    clients = [
        make_client(AsyncMock(return_value=[])),
        make_client(AsyncMock(side_effect=MCPStartupError("server failed"))),
    ]

    with pytest.raises(MCPStartupError, match="server failed"):
        await start_clients(clients)


@pytest.mark.anyio
async def test_start_clients_without_clients_returns_no_tools() -> None:
    assert await start_clients([]) == []
