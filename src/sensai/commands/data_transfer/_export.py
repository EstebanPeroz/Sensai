import json
from pathlib import Path
from typing import TYPE_CHECKING, override
from uuid import UUID

from sensai.commands.base import StateCommand

if TYPE_CHECKING:
    from sensai.history.conversation import Conversation


class Export(StateCommand):
    """Export Command used to export given branch from the db."""

    name = "export"
    help = "Export the current or given branch"
    usage = "/export current | <uuid>"

    _conv: Conversation

    def __init__(self, conv: Conversation) -> None:
        """Init the export Command."""
        super().__init__()
        self._conv = conv

    @override
    def execute(self, *args: str) -> None:
        """Execute a export on the current or given Branch uuid."""
        try:
            (param,) = args
        except ValueError:
            self._ui.send_system_message("Invalid type parameter")
            return
        if param == "current":
            if self._conv.uuid is None:
                self._ui.send_system_message("No conversation to export.")
                return
            self._call_export(self._conv.uuid)
            return
        try:
            uuid = UUID(param)
        except ValueError, TypeError, AttributeError:
            self._ui.send_system_message("Invalid uuid.")
            return
        self._call_export(uuid)

    def _call_export(self, uuid: UUID) -> None:
        export = self._db.export_branch(uuid)
        if export is None:
            self._ui.send_system_message("Failed to export.")
            return
        filename = f"{uuid}.json"
        try:
            with Path(filename).open("w") as file:
                file.write(json.dumps(export))
                self._ui.send_system_message(f"Branch {uuid} exported as {filename}")
        except OSError:
            self._ui.send_system_message(f"Failed to export Branch {uuid}")

    @override
    def completions(self, args: list[str]) -> list[str]:
        """Complete the potential export param."""
        if len(args) != 0:
            return []
        if self._cache_id == []:
            self._fetch_branches()
        return ["current", *(str(uuid) for uuid in self._cache_id)]
