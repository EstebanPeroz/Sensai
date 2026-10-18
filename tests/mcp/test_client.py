import sys
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from sensai.mcp.client import MCPClient
from sensai.mcp.config import MCPServerConfig
from sensai.mcp.discovery import MCPToolInfo

FIXTURE_SERVER = Path(__file__).parent / "fixtures" / "tiny_server.py"


@pytest.mark.anyio
async def test_starts_server_discovers_tools_and_creates_mcp_tools() -> None:
    session = AsyncMock()
    session.list_tools.return_value = [MCPToolInfo("echo", "Echo text", {"type": "object"})]
    config = MCPServerConfig(command="server")
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
    client = MCPClient(MCPServerConfig(command="server", enabled=False))

    with patch("sensai.mcp.client.MCPServerSession") as session_type:
        assert await client.start() == []

    session_type.assert_not_called()


@pytest.mark.anyio
async def test_start_closes_discovery_session() -> None:
    session = AsyncMock()
    session.list_tools.return_value = []
    client = MCPClient(MCPServerConfig(command="server"))

    with patch("sensai.mcp.client.MCPServerSession", return_value=session):
        await client.start()

    session.close.assert_awaited_once()


@pytest.mark.anyio
async def test_start_closes_server_when_tool_discovery_fails() -> None:
    session = AsyncMock()
    session.list_tools.side_effect = RuntimeError("discovery failed")
    client = MCPClient(MCPServerConfig(command="server"))

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
            command=sys.executable,
            args=[str(FIXTURE_SERVER)],
        ),
    )

    try:
        tools = await client.start()
    finally:
        await client.close()

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
