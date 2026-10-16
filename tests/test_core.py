from argparse import Namespace
from unittest.mock import MagicMock, patch

from sensai.config.settings import FeatureSettings, Settings
from sensai.core.core import Core
from sensai.mcp.config import MCPServerConfig
from sensai.mcp.discovery import MCPToolInfo
from sensai.mcp.tool import MCPTool
from sensai.tools.tool_manager import ToolManager


def test_core_does_not_create_mcp_client_without_configuration() -> None:
    provider_manager = MagicMock()
    provider_manager.set_model.return_value = True
    db = MagicMock()
    built_in_tools = [MagicMock()]
    settings = Settings(features=FeatureSettings(disabled=["tools.example"]))

    with patch("sensai.core.core.get_db", return_value=db), patch(
        "sensai.core.core.ProviderManager",
        return_value=provider_manager,
    ), patch(
        "sensai.core.core.MCPClient",
    ) as mcp_client, patch(
        "sensai.core.core.build_tools",
        return_value=built_in_tools,
    ) as build_tools, patch(
        "sensai.core.core.ToolManager",
    ) as tool_manager:
        Core(MagicMock(), settings, Namespace(model="model"))

    mcp_client.assert_not_called()
    build_tools.assert_called_once_with()
    tool_manager.assert_called_once_with(built_in_tools, disabled={"example"})


def test_core_creates_mcp_client_for_configured_server() -> None:
    provider_manager = MagicMock()
    provider_manager.set_model.return_value = True
    config = MCPServerConfig(command="server")
    db = MagicMock()

    with patch("sensai.core.core.get_db", return_value=db), patch(
        "sensai.core.core.ProviderManager",
        return_value=provider_manager,
    ), patch(
        "sensai.core.core.MCPClient",
    ) as mcp_client, patch(
        "sensai.core.core.anyio.run",
        return_value=[],
    ):
        Core(MagicMock(), Settings(mcp=config), Namespace(model="model"))

    mcp_client.assert_called_once_with(config)


def test_core_registers_discovered_mcp_tools() -> None:
    provider_manager = MagicMock()
    provider_manager.set_model.return_value = True
    db = MagicMock()
    config = MCPServerConfig(command="server")
    mcp_tool = MCPTool(MCPToolInfo("echo", "Echo text", {"type": "object"}), MagicMock())

    with patch("sensai.core.core.get_db", return_value=db), patch(
        "sensai.core.core.ProviderManager",
        return_value=provider_manager,
    ), patch(
        "sensai.core.core.MCPClient",
    ) as mcp_client_type, patch(
        "sensai.core.core.anyio.run",
        return_value=[mcp_tool],
    ) as run, patch(
        "sensai.core.core.ToolManager",
    ) as tool_manager_type:
        Core(MagicMock(), Settings(mcp=config), Namespace(model="model"))

    run.assert_called_once_with(mcp_client_type.return_value.start)
    tool_manager_type.assert_called_once()
    registered_tools = tool_manager_type.call_args.args[0]
    assert registered_tools == [mcp_tool]
    assert ToolManager(registered_tools).names() == ["echo"]
