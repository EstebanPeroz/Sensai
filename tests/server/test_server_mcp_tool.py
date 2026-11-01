import asyncio
import os
import signal
import sys
import threading
from collections.abc import Iterator
from concurrent.futures import Future
from pathlib import Path
from unittest.mock import MagicMock

import anyio
import pytest
from sensai_server.config.settings import MCPServerConfig
from sensai_server.tools.mcp import MCPHost, MCPTool
from sensai_server.tools.mcp.discovery import MCPToolInfo
from sensai_server.tools.mcp.errors import MCPCallError
from sensai_server.tools.tool import ToolContext
from sensai_server.tools.tool_manager import ToolManager

from tests.server.fakes import RecordingOutbound

FIXTURE_SERVER = Path(__file__).parents[1] / "mcp" / "fixtures" / "tiny_server.py"


def resolved(value: str) -> Future[str]:
    future: Future[str] = Future()
    future.set_result(value)
    return future


def failed(error: Exception) -> Future[str]:
    future: Future[str] = Future()
    future.set_exception(error)
    return future


def make_tool(
    server_name: str = "server", host: MagicMock | None = None, description: str | None = "Echo a message"
) -> MCPTool:
    return MCPTool(
        MCPToolInfo(
            name="echo",
            description=description,
            input_schema={"type": "object", "properties": {"message": {"type": "string"}}},
        ),
        MagicMock(server_name=server_name),
        host or MagicMock(),
    )


def ctx() -> ToolContext:
    return ToolContext(out=RecordingOutbound())


def make_call(name: str, arguments: dict[str, object] | None = None) -> dict[str, object]:
    return {"function": {"name": name, "arguments": arguments or {}}}


class TestMCPTool:
    def test_name_is_the_discovered_tool_name_prefixed_by_its_server(self) -> None:
        assert make_tool().name == "server__echo"

    def test_same_tool_of_two_servers_gets_two_names(self) -> None:
        first, second = make_tool("first"), make_tool("second")

        assert ToolManager([first, second]).names() == ["first__echo", "second__echo"]

    def test_tool_rule_matches_the_existing_function_schema(self) -> None:
        assert make_tool().tool_rule == {
            "type": "function",
            "function": {
                "name": "server__echo",
                "description": "Echo a message",
                "parameters": {"type": "object", "properties": {"message": {"type": "string"}}},
            },
        }

    def test_missing_description_is_empty(self) -> None:
        assert make_tool(description=None).description == ""

    def test_needs_no_confirmation_by_itself(self) -> None:
        assert not make_tool().requires_confirmation

    async def test_run_calls_the_host_with_the_unprefixed_name_and_returns_text(self) -> None:
        host = MagicMock()
        host.call.return_value = resolved("Echoed message")
        tool = make_tool(host=host)

        assert await tool.run(ctx(), message="hello") == "Echoed message"
        host.call.assert_called_once_with(tool._session, "echo", {"message": "hello"})  # noqa: SLF001

    async def test_run_propagates_mcp_call_errors(self) -> None:
        host = MagicMock()
        host.call.return_value = failed(MCPCallError("tool failed"))

        with pytest.raises(MCPCallError, match="tool failed"):
            await make_tool(host=host).run(ctx(), message="hello")

    async def test_tool_manager_returns_call_errors_to_the_model(self) -> None:
        host = MagicMock()
        host.call.return_value = failed(MCPCallError("tool failed"))
        manager = ToolManager([make_tool(host=host)])

        result = await manager.call(make_call("server__echo", {"message": "hi"}), manager.new_state(), ctx())

        assert result == {"role": "tool", "tool_name": "server__echo", "content": "error: tool failed"}

    async def test_confirmation_is_asked_by_the_tool_manager_when_configured(self) -> None:
        host = MagicMock()
        host.call.return_value = resolved("unused")
        manager = ToolManager([make_tool(host=host)], confirm=["server__echo"])

        assert manager.needs_confirmation(manager.new_state(), "server__echo")


@pytest.fixture
def host() -> Iterator[tuple[MCPHost, ToolManager]]:
    """A host serving the fixture server as `tiny`, and a manager holding its tools."""
    mcp = MCPHost([MCPServerConfig(name="tiny", command=sys.executable, args=[str(FIXTURE_SERVER)], call_timeout=30)])
    tools, failures = mcp.start()
    assert failures == {}
    try:
        yield mcp, ToolManager([*tools])
    finally:
        mcp.close()


def tool_of(manager: ToolManager, name: str) -> MCPTool:
    return manager._tools[name]  # type: ignore[return-value]  # noqa: SLF001


class TestThroughTheHost:
    async def test_tool_manager_runs_the_fixture_tool(self, host: tuple[MCPHost, ToolManager]) -> None:
        _, manager = host

        result = await manager.call(make_call("tiny__echo", {"message": "hello"}), manager.new_state(), ctx())

        assert result == {"role": "tool", "tool_name": "tiny__echo", "content": "hello"}

    async def test_a_disconnected_server_returns_an_error_message(self, host: tuple[MCPHost, ToolManager]) -> None:
        _, manager = host
        state = manager.new_state()
        pid = int((await manager.call(make_call("tiny__pid"), state, ctx()))["content"])
        os.kill(pid, signal.SIGKILL)

        results = [
            (await manager.call(make_call("tiny__echo", {"message": "gone"}), state, ctx()))["content"]
            for _ in range(2)
        ]

        for content in results:
            assert content.startswith("error: MCP server 'tiny' disconnected, restart the server to reconnect")

    async def test_cancelling_the_calls_cancels_the_mcp_call(self, host: tuple[MCPHost, ToolManager]) -> None:
        _, manager = host
        session = tool_of(manager, "tiny__slow")._session  # noqa: SLF001
        started, cancelled = threading.Event(), threading.Event()
        call_tool = session.call_tool

        async def recording_call_tool(name: str, arguments: dict[str, object]) -> str:
            started.set()
            try:
                return await call_tool(name, arguments)
            except anyio.get_cancelled_exc_class():
                cancelled.set()
                raise

        session.call_tool = recording_call_tool  # type: ignore[method-assign]
        state = manager.new_state()
        calls = asyncio.create_task(manager.calls([make_call("tiny__slow", {"seconds": 30})], state, ctx()))
        assert await asyncio.to_thread(started.wait, 5)

        calls.cancel()
        with pytest.raises(asyncio.CancelledError):
            await calls

        assert await asyncio.to_thread(cancelled.wait, 5)
        # The session survived the cancelled call.
        result = await manager.call(make_call("tiny__echo", {"message": "after"}), state, ctx())
        assert result["content"] == "after"
