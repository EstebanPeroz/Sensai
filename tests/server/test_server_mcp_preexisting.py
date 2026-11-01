import asyncio
import shutil
from pathlib import Path

import pytest
from sensai_server.config.settings import MCPServerConfig
from sensai_server.tools.mcp import MCPHost
from sensai_server.tools.tool import ToolContext
from sensai_server.tools.tool_manager import ToolManager

from tests.server.fakes import RecordingOutbound


async def test_filesystem_server_tool_runs_through_the_host(tmp_path: Path) -> None:
    npx = shutil.which("npx")
    if npx is None:
        pytest.skip("npx is required for the pre-existing MCP server demonstration")

    host = MCPHost(
        [
            MCPServerConfig(
                name="server",
                command=npx,
                args=["--yes", "@modelcontextprotocol/server-filesystem", str(tmp_path)],
                startup_timeout=30,
                call_timeout=30,
            ),
        ],
    )
    tools, failures = await asyncio.to_thread(host.start)
    try:
        manager = ToolManager([*tools])
        result = await manager.call(
            {"function": {"name": "server__list_allowed_directories", "arguments": {}}},
            manager.new_state(),
            ToolContext(out=RecordingOutbound()),
        )
    finally:
        await asyncio.to_thread(host.close)

    assert failures == {}
    assert result["role"] == "tool"
    assert result["tool_name"] == "server__list_allowed_directories"
    assert str(tmp_path) in result["content"]
