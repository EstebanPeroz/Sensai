import socket
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import anyio
import pytest

from sensai.mcp.client import MCPClient, start_clients
from sensai.mcp.config import MCPServerConfig
from sensai.mcp.discovery import MCPToolInfo
from sensai.mcp.errors import MCPStartupError
from sensai.tools.tool_manager import ToolManager

FIXTURE_SERVER = Path(__file__).parent / "fixtures" / "tiny_server.py"


@pytest.mark.anyio
async def test_starts_server_discovers_tools_and_creates_mcp_tools() -> None:
    session = AsyncMock(server_name="server")
    session.list_tools.return_value = [MCPToolInfo("echo", "Echo text", {"type": "object"})]
    config = MCPServerConfig(name="server", command="server")
    client = MCPClient(config)

    with patch("sensai.mcp.client.MCPServerSession", return_value=session) as session_type:
        tools = await client.start()

    session_type.assert_called_once_with(config)
    session.start.assert_awaited_once()
    session.list_tools.assert_awaited_once()
    assert len(tools) == 1
    assert tools[0].name == "server__echo"


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
            MCPStartupError,
            match="discovery failed",
        ),
    ):
        await client.start()

    session.close.assert_awaited_once()


@pytest.mark.anyio
async def test_start_gives_up_when_tool_discovery_exceeds_startup_timeout() -> None:
    session = AsyncMock()
    session.list_tools.side_effect = anyio.sleep_forever
    client = MCPClient(MCPServerConfig(name="server", command="server", startup_timeout=0.01))

    with (
        patch("sensai.mcp.client.MCPServerSession", return_value=session),
        anyio.fail_after(1),
        pytest.raises(MCPStartupError, match=r"no answer within 0\.01s"),
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
    assert tools[0].name == "server__echo"
    assert tools[0].tool_rule == {
        "type": "function",
        "function": {
            "name": "server__echo",
            "description": "Echo a message",
            "parameters": {
                "properties": {"message": {"title": "Message", "type": "string"}},
                "required": ["message"],
                "title": "echoArguments",
                "type": "object",
            },
        },
    }


def make_client(start: AsyncMock, name: str = "server") -> MagicMock:
    client = MagicMock(spec=MCPClient)
    client.name = name
    client.start = start
    return client


@pytest.mark.anyio
async def test_start_clients_returns_tools_in_client_order() -> None:
    first_tool, second_tool = MagicMock(), MagicMock()

    async def slow_start() -> list[MagicMock]:
        await anyio.sleep(0.01)
        return [first_tool]

    clients = [make_client(AsyncMock(side_effect=slow_start)), make_client(AsyncMock(return_value=[second_tool]))]

    assert await start_clients(clients) == ([first_tool, second_tool], {})


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
        assert await start_clients(clients) == ([], {})


@pytest.mark.anyio
async def test_start_clients_skips_failed_clients_and_reports_them_by_name() -> None:
    tool = MagicMock()
    error = MCPStartupError("server failed")
    clients = [
        make_client(AsyncMock(side_effect=error), name="broken"),
        make_client(AsyncMock(return_value=[tool]), name="working"),
    ]

    assert await start_clients(clients) == ([tool], {"broken": error})


@pytest.mark.anyio
async def test_start_clients_starts_fixture_next_to_a_missing_server() -> None:
    clients = [
        MCPClient(MCPServerConfig(name="missing", command="sensai-missing-mcp-server")),
        MCPClient(MCPServerConfig(name="fixture", command=sys.executable, args=[str(FIXTURE_SERVER)])),
    ]

    tools, failures = await start_clients(clients)

    assert [tool.name for tool in tools] == ["fixture__echo"]
    assert list(failures) == ["missing"]
    assert isinstance(failures["missing"], MCPStartupError)


@pytest.mark.anyio
async def test_start_clients_without_clients_returns_no_tools() -> None:
    assert await start_clients([]) == ([], {})


@pytest.fixture
def http_fixture_url() -> Iterator[str]:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    server = subprocess.Popen(  # noqa: S603
        [sys.executable, str(FIXTURE_SERVER), "--http", str(port)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        anyio.run(_wait_for_port, port)
        yield f"http://127.0.0.1:{port}/mcp"
    finally:
        server.terminate()
        server.wait()


async def _wait_for_port(port: int) -> None:
    with anyio.fail_after(10):
        while True:
            try:
                stream = await anyio.connect_tcp("127.0.0.1", port)
            except OSError:
                await anyio.sleep(0.05)
            else:
                await stream.aclose()
                return


@pytest.mark.anyio
async def test_discovers_and_calls_fixture_tool_over_http(http_fixture_url: str) -> None:
    client = MCPClient(MCPServerConfig(name="remote", transport="http", url=http_fixture_url))

    tools = await client.start()
    result = await anyio.to_thread.run_sync(
        ToolManager(tools).call,
        {"function": {"name": "remote__echo", "arguments": {"message": "hello over http"}}},
    )

    assert [tool.name for tool in tools] == ["remote__echo"]
    assert result == {"role": "tool", "tool_name": "remote__echo", "content": "hello over http"}
