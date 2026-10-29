from __future__ import annotations

import argparse
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

# The server only listens on loopback: the client must run on the same machine.
HOST = "127.0.0.1"
DEFAULT_PORT = 8000


def server_url(port: int) -> str:
    """Return the websocket URL of the local server listening on `port`."""
    return f"ws://{HOST}:{port}/ws"


class Parsing:
    """Command line arguments parser."""

    _parser: argparse.ArgumentParser

    def __init__(self) -> None:
        """Declare the command line arguments."""
        self._parser = argparse.ArgumentParser(prog="sensai-tui", description="Terminal client of a Sensai server.")
        self._parser.add_argument(
            "--port", type=int, default=DEFAULT_PORT, help=f"port of the local server (default: {DEFAULT_PORT})"
        )
        self._parser.add_argument("--model", required=True, help="model to log in with")

    def parse_args(self, argv: Sequence[str] | None = None) -> argparse.Namespace:
        """Parse the command line arguments."""
        return self._parser.parse_args(argv)
