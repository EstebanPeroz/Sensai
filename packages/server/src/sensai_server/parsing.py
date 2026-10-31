from __future__ import annotations

import argparse
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

# The server only listens on loopback: it is a local daemon, reachable from this machine only.
HOST = "127.0.0.1"
DEFAULT_PORT = 8000


class Parsing:
    """Command line arguments parser."""

    _parser: argparse.ArgumentParser

    def __init__(self) -> None:
        """Declare the command line arguments."""
        self._parser = argparse.ArgumentParser(prog="sensai-server", description="Sensai server, over a websocket.")
        self._parser.add_argument(
            "--port", type=int, default=DEFAULT_PORT, help=f"port to listen on (default: {DEFAULT_PORT})"
        )
        self._parser.add_argument("--config", type=Path, metavar="PATH", required=True, help="settings file to load")

    def parse_args(self, argv: Sequence[str] | None = None) -> argparse.Namespace:
        """Parse the command line arguments."""
        return self._parser.parse_args(argv)
