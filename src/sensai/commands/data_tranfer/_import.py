import json
from pathlib import Path
from typing import override

from sensai.commands.base import StateCommand
from sensai.memory.persistent_db.error import DatabaseError


class Import(StateCommand):
    """Import Command used to create a branch from an exported json file."""

    name = "import"
    help = "Import a branch from an exported json file"
    usage = "/import <path>"

    @override
    def execute(self, *args: str) -> None:
        """Import the branch described by the json file at the given path."""
        try:
            (path,) = args
        except ValueError:
            self._ui.send_system_message(f"Invalid arguments. Usage: {self.usage}")
            return
        config = self._load_json_dict(path)
        if config is None:
            return
        try:
            self._db.import_branch(config)
        except DatabaseError as err:
            self._ui.send_system_message(err.message)
            return
        self._cache_id.clear()
        self._ui.send_system_message(f"Branch imported from {path}")

    def _load_json_dict(self, path: str) -> dict | None:
        try:
            with Path(path).open() as json_file:
                data = json.load(json_file)
        except OSError as e:
            self._ui.send_system_message(f"Cannot read {path}: {e.strerror}")
            return None
        except json.JSONDecodeError as e:
            self._ui.send_system_message(f"Invalid JSON in {path}: {e}")
            return None
        if not isinstance(data, dict):
            self._ui.send_system_message(f"Expected a JSON object in {path}, got {type(data).__name__}")
            return None
        return data
