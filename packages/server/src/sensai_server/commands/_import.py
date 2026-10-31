from __future__ import annotations

import asyncio
import json
from typing import TYPE_CHECKING, cast, override

from sensai_protocol import ClientCommand, FileContent, ReadFileParams

from sensai_server.app.outbound import ClientCommandError
from sensai_server.commands._base import Command
from sensai_server.memory.persistent_db.error import DatabaseError

if TYPE_CHECKING:
    from sensai_server.memory.persistent_db.adapter import PersistentDatabase
    from sensai_server.session import Session


class Import(Command):
    """Import command class to create a branch from a JSON file on the client's machine.

    The client reads the file and the server parses it: the file belongs to the client's machine, not the server's.
    """

    name = "import"
    help = "Import a branch from an exported JSON file."
    usage = "/import <path>"
    _db: PersistentDatabase

    def __init__(self, db: PersistentDatabase) -> None:
        """Initialize the Import command on the database `db`."""
        super().__init__()
        self._db = db

    @override
    async def execute(self, session: Session, *args: str) -> None:
        """Import the branch described by the JSON file at the path given as the only argument."""
        try:
            (path,) = args
        except ValueError:
            await session.out.send_system_message(f"Invalid arguments. Usage: {self.usage}")
            return
        config = await self._load_json_dict(session, path)
        if config is None:
            return
        try:
            await asyncio.shield(asyncio.to_thread(self._db.import_branch, config))
        except DatabaseError as err:
            await session.out.send_system_message(err.message)
            return
        await session.out.send_system_message(f"Branch imported from {path}")

    @staticmethod
    async def _load_json_dict(session: Session, path: str) -> dict | None:
        """Read the file at `path` on the client and parse it as a JSON object, None when it fails (reported)."""
        try:
            file = await session.out.request_command(ClientCommand.READ_FILE, ReadFileParams(path=path))
        except ClientCommandError as err:
            await session.out.send_system_message(f"Cannot read {path}: {err.reason}")
            return None
        try:
            data = json.loads(cast("FileContent", file).content)
        except json.JSONDecodeError as err:
            await session.out.send_system_message(f"Invalid JSON in {path}: {err}")
            return None
        if not isinstance(data, dict):
            await session.out.send_system_message(f"Expected a JSON object in {path}, got {type(data).__name__}")
            return None
        return data
