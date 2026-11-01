import pytest
from sensai_server.tools.error import DuplicateToolError
from sensai_server.tools.state import ToolState
from sensai_server.tools.tool import Tool, ToolContext
from sensai_server.tools.tool_manager import ToolManager


class _NamedTool(Tool[str]):
    def __init__(self, name: str) -> None:
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return "Do nothing."

    @property
    def parameters(self) -> dict:
        return {"type": "object", "properties": {}}

    async def run(self, ctx: ToolContext, **kwargs: object) -> str:
        del ctx, kwargs
        return ""


class TestNewState:
    def test_every_tool_enabled_by_default(self) -> None:
        manager = ToolManager([_NamedTool("a"), _NamedTool("b")])
        assert manager.new_state() == ToolState(enabled={"a", "b"}, always_allowed=set())

    def test_disabled_tools_left_out(self) -> None:
        manager = ToolManager([_NamedTool("a"), _NamedTool("b")], disabled=["b", "ghost"])
        assert manager.new_state().enabled == {"a"}
        assert manager.unknown_disabled() == ["ghost"]

    def test_each_state_is_fresh(self) -> None:
        manager = ToolManager([_NamedTool("a")])
        first, second = manager.new_state(), manager.new_state()
        first.enabled.discard("a")
        first.always_allowed.add("a")
        assert second == ToolState(enabled={"a"}, always_allowed=set())

    def test_unknown_confirm_reported(self) -> None:
        manager = ToolManager([_NamedTool("a")], confirm=["a", "ghost"])
        assert manager.unknown_confirm() == ["ghost"]

    def test_duplicate_names_rejected(self) -> None:
        with pytest.raises(DuplicateToolError):
            ToolManager([_NamedTool("a"), _NamedTool("a")])
