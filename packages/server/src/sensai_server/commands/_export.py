from __future__ import annotations

import asyncio
import json
from typing import TYPE_CHECKING, cast, override
from uuid import UUID

from sensai_protocol import MAX_FILE_SIZE, ClientCommand, FileWritten, WriteFileParams

from sensai_server.app.outbound import ClientCommandError
from sensai_server.commands._base import Command

if TYPE_CHECKING:
    from sensai_server.memory.persistent_db.adapter import PersistentDatabase
    from sensai_server.session import Session

_GIVE_PATH = 2


class Export(Command):
    """Export command class to save a branch as a JSON file on the client's machine.

    The server builds the JSON and the client writes it: the file belongs to the client's machine, not the server's.
    """

    name = "export"
    help = "Export the current or given branch as a JSON file."
    usage = "/export current | <uuid> [path]"
    _db: PersistentDatabase

    def __init__(self, db: PersistentDatabase) -> None:
        """Initialize the Export command on the database `db`."""
        super().__init__()
        self._db = db

    @override
    async def execute(self, session: Session, *args: str) -> None:
        """Export the branch named by the first argument to the path given by the second, `<uuid>.json` if none."""
        if len(args) not in [1, 2]:
            await session.out.send_system_message(f"Usage: {self.usage}")
            return
        uuid = await self._get_uuid(args[0], session)
        if uuid is None:
            return
        export = await asyncio.to_thread(self._db.export_branch, uuid)
        if export is None:
            await session.out.send_system_message(f"Branch {uuid} not found.")
            return
        content = json.dumps(export, indent=2)
        if len(content) > MAX_FILE_SIZE:
            await session.out.send_system_message(f"Failed to export branch {uuid}: too large to send to the client.")
            return
        path = args[1] if len(args) == _GIVE_PATH else f"{uuid}.json"
        try:
            written = await session.out.request_command(
                ClientCommand.WRITE_FILE, WriteFileParams(path=path, content=content)
            )
        except ClientCommandError as err:
            await session.out.send_system_message(f"Failed to export branch {uuid}: {err.reason}")
            return
        await session.out.send_system_message(f"Branch {uuid} exported to {cast('FileWritten', written).path}")

    @override
    def completions(self, session: Session, args: list[str]) -> list[str]:
        """Return `current` and the uuids of the branches for the first argument, nothing after."""
        del session
        if args:
            return []
        return ["current", *map(str, self._db.get_branches())]

    async def _get_uuid(self, target: str, session: Session) -> UUID | None:
        if target == "current":
            uuid = session.conv.uuid
            if uuid is None:
                await session.out.send_system_message("No conversation to export.")
                return None
        else:
            uuid = self._try_parse_uuid(target)
            if uuid is None:
                await session.out.send_system_message("Invalid uuid.")
                return None
        return uuid

    @staticmethod
    def _try_parse_uuid(value: str) -> UUID | None:
        try:
            return UUID(value)
        except ValueError, TypeError, AttributeError:
            return None
