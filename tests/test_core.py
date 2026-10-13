from argparse import Namespace
from unittest.mock import MagicMock, patch

from sensai.config.settings import Settings
from sensai.core.core import Core
from sensai.mcp.config import MCPServerConfig


def test_core_does_not_create_mcp_client_without_configuration() -> None:
    provider_manager = MagicMock()
    provider_manager.set_model.return_value = True

    with patch("sensai.core.core.ProviderManager", return_value=provider_manager), patch(
        "sensai.core.core.MCPClient",
    ) as mcp_client:
        Core(MagicMock(), Settings(), Namespace(model="model"))

    mcp_client.assert_not_called()


def test_core_creates_mcp_client_for_configured_server() -> None:
    provider_manager = MagicMock()
    provider_manager.set_model.return_value = True
    config = MCPServerConfig(command="server")

    with patch("sensai.core.core.ProviderManager", return_value=provider_manager), patch(
        "sensai.core.core.MCPClient",
    ) as mcp_client:
        Core(MagicMock(), Settings(mcp=config), Namespace(model="model"))

    mcp_client.assert_called_once_with(config)
