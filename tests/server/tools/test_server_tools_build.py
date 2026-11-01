"""Invariants of the tool composition root.

They fail the day a tool is added with a name clashing with another one, or with a
`tool_rule` advertising a name the `ToolManager` cannot look up.
"""

from collections.abc import Iterator

import pytest
from sensai_server.config.settings import Settings
from sensai_server.tools import BuiltTools, build_tools
from sensai_server.tools.tool import Tool
from sensai_server.tools.tool_manager import ToolManager

from tests.server.fakes import FakeEmbedder


@pytest.fixture
def built() -> Iterator[BuiltTools]:
    """The tools built from the default settings, with an embedder, the MCP host closed after the test."""
    built = build_tools(Settings(), FakeEmbedder())
    yield built
    built.mcp.close()


def _names(built: BuiltTools) -> list[str]:
    return [tool.name for tool in built.tools]


class TestBuildTools:
    def test_every_built_object_is_a_tool(self, built: BuiltTools) -> None:
        assert all(isinstance(tool, Tool) for tool in built.tools)

    def test_every_tool_has_a_non_empty_name(self, built: BuiltTools) -> None:
        assert all(tool.name for tool in built.tools)

    def test_tool_names_are_unique(self, built: BuiltTools) -> None:
        names = _names(built)

        assert sorted(names) == sorted(set(names))

    def test_each_rule_advertises_the_name_the_manager_registers(self, built: BuiltTools) -> None:
        # A rule naming anything else would let the model call a name `ToolManager.calls` cannot resolve.
        for tool in built.tools:
            assert tool.tool_rule["function"]["name"] == tool.name

    def test_each_call_returns_a_fresh_list(self, built: BuiltTools) -> None:
        other = build_tools(Settings(), FakeEmbedder())
        try:
            assert other.tools is not built.tools
        finally:
            other.mcp.close()

    def test_builds_the_web_tools_with_an_embedder(self, built: BuiltTools) -> None:
        assert sorted(_names(built)) == ["ask_user", "web_fetch", "web_search"]

    def test_leaves_the_web_tools_out_without_an_embedder(self) -> None:
        built = build_tools(Settings(), None)
        try:
            assert _names(built) == ["ask_user"]
        finally:
            built.mcp.close()


class TestBuildToolsFeedsTheManager:
    def test_the_manager_accepts_the_built_tools(self, built: BuiltTools) -> None:
        manager = ToolManager(built.tools)

        assert manager.names() == sorted(_names(built))

    def test_every_built_tool_starts_enabled(self, built: BuiltTools) -> None:
        manager = ToolManager(built.tools)
        state = manager.new_state()

        assert all(manager.is_enabled(state, name) for name in manager.names())

    def test_every_built_tool_is_advertised_to_the_model(self, built: BuiltTools) -> None:
        manager = ToolManager(built.tools)

        assert len(manager.get_tool_call_rules(manager.new_state())) == len(built.tools)

    def test_built_tools_can_all_be_disabled_by_name(self, built: BuiltTools) -> None:
        manager = ToolManager(built.tools)
        state = manager.new_state()

        assert all(manager.disable(state, name) for name in manager.names())
        assert manager.get_tool_call_rules(state) == []
