from __future__ import annotations

import asyncio
import sys
from typing import TYPE_CHECKING

from sensai_tui.connection import Connection, ConnectionFailedError
from sensai_tui.parsing import Parsing
from sensai_tui.ui import ChatApp

if TYPE_CHECKING:
    from collections.abc import Sequence


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point: connect to the server and run the terminal client."""
    args = Parsing().parse_args(argv)
    return asyncio.run(_run(Connection(args.url), args.model))


async def _run(connection: Connection, model: str) -> int:
    """Connect to the server, then run the chat UI on the open connection. Return the exit code."""
    try:
        await connection.open()
    except ConnectionFailedError as err:
        print(f"sensai-tui: cannot connect to {connection.url}: {err}", file=sys.stderr)
        return 1
    await ChatApp(connection, model).run_async()
    return 0
