import shutil
from pathlib import Path

import pytest
from anyio import to_thread

from sensai.mcp.client import MCPClient
from sensai.mcp.config import MCPServerConfig
from sensai.tools.tool import Tool
from sensai.tools.tool_manager import ToolManager


@pytest.mark.anyio
async def test_filesystem_server_tool_runs_through_host_path(tmp_path: Path) -> None:
    npx = shutil.which("npx")
    if npx is None:
        pytest.skip("npx is required for the pre-existing MCP server demonstration")

    client = MCPClient(
        MCPServerConfig(
            name="server",
            command=npx,
            args=["--yes", "@modelcontextprotocol/server-filesystem", str(tmp_path)],
            startup_timeout=30,
            call_timeout=30,
        ),
    )

    tools = await client.start()
    tool: Tool[object] = next(tool for tool in tools if tool.name == "list_allowed_directories")
    manager = ToolManager([tool])
    result = await to_thread.run_sync(
        manager.call,
        {
            "type": "function",
            "function": {"name": "list_allowed_directories", "arguments": {}},
        },
    )

    assert result is not None
    assert result["role"] == "tool"
    assert result["tool_name"] == "list_allowed_directories"
    assert str(tmp_path) in result["content"]
