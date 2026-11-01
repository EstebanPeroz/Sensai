import json
import logging
import os
import sys
import threading
from pathlib import Path

import pytest
import sensai_server
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sensai_protocol import (
    AiChunk,
    Chat,
    Command,
    ContextUsage,
    Input,
    Login,
    Role,
    ServerMessage,
    Session,
    System,
    parse_server_message,
)
from sensai_server import main
from sensai_server.app import app as app_module
from sensai_server.app import create_app
from sensai_server.app._state import app_state
from sensai_server.config.settings import MCPServerConfig, Settings
from sensai_server.memory.persistent_db.error import DbConnectionError
from sensai_server.tools import build_tools
from sensai_server.tools.mcp import MCPTool
from sensai_server.tools.tool_manager import DuplicateToolError
from starlette.testclient import WebSocketTestSession

from tests.server.conftest import MODEL, SETTINGS, settings_with_db
from tests.server.fakes import FakeAdapter, call, chunk, tool_call_chunk

FIXTURE_SERVER = Path(__file__).parents[1] / "tools" / "mcp" / "fixtures" / "tiny_server.py"


def tiny(name: str = "tiny") -> MCPServerConfig:
    return MCPServerConfig(name=name, command=sys.executable, args=[str(FIXTURE_SERVER)])


def broken(name: str = "broken") -> MCPServerConfig:
    return MCPServerConfig(name=name, command="/nonexistent/mcp-server", startup_timeout=5)


def with_mcp(tmp_path: Path, *configs: MCPServerConfig) -> Settings:
    settings = settings_with_db(tmp_path)
    return Settings(llm=settings.llm, db=settings.db, mcp=list(configs))


def threads() -> set[threading.Thread]:
    """The live threads. Compare with `<=`: a thread of an earlier test may end in between, never a new one start."""
    return {thread for thread in threading.enumerate() if thread.is_alive()}


def is_running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def receive(ws: WebSocketTestSession) -> ServerMessage:
    return parse_server_message(ws.receive_text())


def login(ws: WebSocketTestSession) -> None:
    ws.send_text(Login(model=MODEL).model_dump_json())
    assert isinstance(receive(ws), Session)
    assert isinstance(receive(ws), ContextUsage)


def tiny_pid(app: FastAPI) -> int:
    """The pid of the `tiny` server's subprocess, asked through the host as `MCPTool.run` does."""
    tool = app_state(app).tool_manager._tools["tiny__pid"]  # noqa: SLF001
    assert isinstance(tool, MCPTool)
    return int(app_state(app).mcp.call(tool._session, "pid", {}).result(timeout=10))  # noqa: SLF001


class TestBuildTools:
    def test_appends_the_mcp_tools_after_the_built_in_ones(self) -> None:
        built = build_tools(Settings(llm=SETTINGS.llm, mcp=[tiny()]), None)
        try:
            names = [tool.name for tool in built.tools]
            assert names[0] == "ask_user"
            assert "tiny__echo" in names
        finally:
            built.mcp.close()

    def test_no_mcp_server_starts_no_thread(self) -> None:
        before = threads()
        built = build_tools(SETTINGS, None)
        assert threads() <= before
        assert [tool.name for tool in built.tools] == ["ask_user"]
        built.mcp.close()

    def test_failing_server_is_logged_and_left_out(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.WARNING, logger="sensai_server.tools"):
            built = build_tools(Settings(llm=SETTINGS.llm, mcp=[broken(), tiny()]), None)
        try:
            names = [tool.name for tool in built.tools]
            assert "tiny__echo" in names
            assert not any(name.startswith("broken__") for name in names)
            assert "MCP server 'broken' failed to start, its tools are unavailable" in caplog.text
        finally:
            built.mcp.close()


class TestApp:
    def test_mcp_tool_is_listed_by_the_tools_command(self, tmp_path: Path) -> None:
        with TestClient(create_app(with_mcp(tmp_path, tiny()))) as client, client.websocket_connect("/ws") as ws:
            login(ws)
            ws.send_text(Command(command="/tools").model_dump_json())
            assert receive(ws) == Input(role=Role.USER, content="/tools", command=True)
            listing = receive(ws)
            assert isinstance(listing, System)
            assert "- tiny__echo: enabled" in listing.message

    def test_model_calls_the_mcp_tool_through_ws(self, tmp_path: Path) -> None:
        FakeAdapter.tools = True
        FakeAdapter.script = [
            [chunk("I should echo.", done=True)],
            [tool_call_chunk(call("tiny__echo", {"message": "pong"})), chunk("", done=True)],
        ]
        FakeAdapter.replies = [chunk("It said pong.", done=True)]
        with TestClient(create_app(with_mcp(tmp_path, tiny()))) as client, client.websocket_connect("/ws") as ws:
            login(ws)
            ws.send_text(Chat(prompt="echo pong").model_dump_json())
            while receive(ws) != AiChunk(content="It said pong.", done=True):
                pass
        tool_messages = [m for payload in FakeAdapter.payloads for m in payload["messages"] if m["role"] == "tool"]
        assert {"role": "tool", "tool_name": "tiny__echo", "content": "pong"} in tool_messages

    def test_failing_server_still_lets_the_app_start(self, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.WARNING):
            app = create_app(with_mcp(tmp_path, broken()))
        with TestClient(app) as client, client.websocket_connect("/ws") as ws:
            login(ws)
        assert "MCP server 'broken' failed to start" in caplog.text
        assert app_state(app).tool_manager.names() == ["ask_user"]

    def test_stopping_the_app_leaves_no_subprocess_and_no_thread(self, tmp_path: Path) -> None:
        before = threads()
        app = create_app(with_mcp(tmp_path, tiny()))
        with TestClient(app):
            pid = tiny_pid(app)
            assert is_running(pid)
        assert not is_running(pid)
        assert threads() <= before

    def test_a_later_failure_closes_the_host(self) -> None:
        before = threads()
        with pytest.raises(DbConnectionError):
            create_app(Settings(llm=SETTINGS.llm, mcp=[tiny()]))  # no persistent database
        assert threads() <= before

    def test_a_failing_tool_manager_closes_the_host(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        def refuse(*_args: object, **_kwargs: object) -> None:
            msg = "duplicate tool"
            raise DuplicateToolError(msg)

        monkeypatch.setattr(app_module, "ToolManager", refuse)
        before = threads()
        with pytest.raises(DuplicateToolError):
            create_app(with_mcp(tmp_path, tiny()))
        assert threads() <= before


class TestMain:
    def test_closes_the_host_when_uvicorn_returns(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        apps: list[FastAPI] = []
        monkeypatch.setattr(sensai_server.uvicorn, "run", lambda app, **_kwargs: apps.append(app))
        path = tmp_path / "settings.toml"
        path.write_text(
            f'[llm.ollama]\nbase_url = "http://ollama:1234"\n[db.persistent]\nurl = "sqlite:///{tmp_path / "t.db"}"\n'
            f"[mcp.tiny]\ncommand = {json.dumps(sys.executable)}\nargs = [{json.dumps(str(FIXTURE_SERVER))}]\n"
        )
        before = threads()

        assert main(["--config", str(path)]) == 0

        [app] = apps
        assert "tiny__echo" in app_state(app).tool_manager.names()
        assert threads() <= before
