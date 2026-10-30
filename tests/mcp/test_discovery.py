from unittest.mock import patch

import pytest

from sensai.mcp.config import MCPServerConfig
from sensai.mcp.discovery import MCPToolInfo
from sensai.mcp.session import MCPServerSession
from tests.mcp.test_session import AsyncContext, FakeClientSession


@pytest.mark.anyio
async def test_reads_tool_name_description_and_input_schema() -> None:
    stdio = AsyncContext(("read-stream", "write-stream"))
    client = FakeClientSession()
    client.tools = [
        type(
            "Tool",
            (),
            {
                "name": "echo",
                "description": "Echo a message",
                "input_schema": {"type": "object", "properties": {"message": {"type": "string"}}},
            },
        )(),
    ]
    session = MCPServerSession(MCPServerConfig(name="server", command="fake-server"))

    with (
        patch("sensai.mcp.session.stdio_client", return_value=stdio),
        patch("sensai.mcp.session.ClientSession", return_value=AsyncContext(client)),
    ):
        await session.start()
        tools = await session.list_tools()

    assert tools == [
        MCPToolInfo(
            name="echo",
            description="Echo a message",
            input_schema={"type": "object", "properties": {"message": {"type": "string"}}},
        ),
    ]
