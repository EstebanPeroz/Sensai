"""Invariants of the tool composition root.

They fail the day a tool is added with a name clashing with another one, or with a
`tool_rule` advertising a name the `ToolManager` cannot look up.
"""

from sensai.config.settings import Settings
from sensai.llm.embedder import Embedder
from sensai.tools import build_tools as build_tools_from
from sensai.tools.tool import Tool
from sensai.tools.tool_manager import ToolManager


class FakeEmbedder(Embedder):
    def __init__(self) -> None:
        pass

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[1.0] for _ in texts]


def no_answer(_question: str) -> str | None:
    return None


def build_tools() -> list[Tool]:
    return build_tools_from(Settings(), FakeEmbedder(), no_answer)


class TestBuildTools:
    def test_returns_a_list(self) -> None:
        assert isinstance(build_tools(), list)

    def test_every_built_object_is_a_tool(self) -> None:
        assert all(isinstance(tool, Tool) for tool in build_tools())

    def test_every_tool_has_a_non_empty_name(self) -> None:
        assert all(tool.name for tool in build_tools())

    def test_tool_names_are_unique(self) -> None:
        names = [tool.name for tool in build_tools()]

        assert sorted(names) == sorted(set(names))

    def test_each_rule_advertises_the_name_the_manager_registers(self) -> None:
        # A rule naming anything else would let the model call a name `ToolManager.call` cannot resolve.
        for tool in build_tools():
            assert tool.tool_rule["function"]["name"] == tool.name

    def test_each_call_returns_a_fresh_list(self) -> None:
        assert build_tools() is not build_tools()

    def test_builds_the_web_tools_with_an_embedder(self) -> None:
        assert sorted(tool.name for tool in build_tools()) == ["ask_user", "web_fetch", "web_search"]

    def test_leaves_the_web_tools_out_without_an_embedder(self) -> None:
        assert [tool.name for tool in build_tools_from(Settings(), None, no_answer)] == ["ask_user"]

    def test_ask_user_asks_through_the_given_callback(self) -> None:
        questions: list[str] = []

        def ask(question: str) -> str:
            questions.append(question)
            return "Q3"

        ask_user = next(tool for tool in build_tools_from(Settings(), None, ask) if tool.name == "ask_user")

        assert ask_user.run(question="Which quarter?") == "Q3"
        assert questions == ["Which quarter?"]


class TestBuildToolsFeedsTheManager:
    def test_the_manager_accepts_the_built_tools(self) -> None:
        manager = ToolManager(build_tools())

        assert manager.names() == sorted(tool.name for tool in build_tools())

    def test_every_built_tool_starts_enabled(self) -> None:
        manager = ToolManager(build_tools())

        assert all(manager.is_enabled(name) for name in manager.names())

    def test_every_built_tool_is_advertised_to_the_model(self) -> None:
        manager = ToolManager(build_tools())

        assert len(manager.get_tool_call_rules()) == len(build_tools())

    def test_built_tools_can_all_be_disabled_by_name(self) -> None:
        manager = ToolManager(build_tools())

        assert all(manager.disable(name) for name in manager.names())
        assert manager.get_tool_call_rules() == []
