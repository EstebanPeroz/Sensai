from __future__ import annotations

from typing import TYPE_CHECKING

from sensai_server.tools.error import DuplicateToolError
from sensai_server.tools.state import ToolState

if TYPE_CHECKING:
    from collections.abc import Iterable

    from sensai_server.tools.tool import Tool


class ToolManager:
    """Registry of the available tools, shared by every session.

    It keeps no per-client state: each session holds its own `ToolState`, built by `new_state`.
    """

    _tools: dict[str, Tool]
    _disabled: frozenset[str]
    _unknown_disabled: frozenset[str]
    _confirm: frozenset[str]
    _unknown_confirm: frozenset[str]

    def __init__(
        self,
        tools: list[Tool] | None = None,
        disabled: Iterable[str] | None = None,
        *,
        confirm: Iterable[str] | None = None,
    ) -> None:
        """Register `tools`; a new session enables all of them but the ones named in `disabled`.

        The tools named in `confirm`, and those requiring it themselves, run only once the user allows them.

        Names in `disabled` or `confirm` matching no registered tool are ignored; `unknown_disabled()`
        and `unknown_confirm()` report them so the caller can warn about a stale or misspelled name.

        Raises:
            DuplicateToolError: If two tools share the same name.

        """
        self._tools = {}
        for tool in tools or []:
            if tool.name in self._tools:
                raise DuplicateToolError(tool.name)
            self._tools[tool.name] = tool
        requested = frozenset(disabled or ())
        self._disabled = requested & set(self._tools)
        self._unknown_disabled = requested - set(self._tools)
        self._confirm = frozenset(confirm or ())
        self._unknown_confirm = self._confirm - set(self._tools)

    def names(self) -> list[str]:
        """Return the name of every registered tool, enabled or not, sorted."""
        return sorted(self._tools)

    def unknown_disabled(self) -> list[str]:
        """Return the names given as disabled that match no registered tool, sorted."""
        return sorted(self._unknown_disabled)

    def unknown_confirm(self) -> list[str]:
        """Return the names given as needing confirmation that match no registered tool, sorted."""
        return sorted(self._unknown_confirm)

    def new_state(self) -> ToolState:
        """Return the tool state of a new session: every tool enabled but the disabled ones, none always allowed."""
        return ToolState(enabled=set(self._tools) - self._disabled)
