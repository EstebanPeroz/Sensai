from __future__ import annotations

import argparse
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

    from pydantic_core.core_schema import ArgumentsParameter

DEFAULT_URL = "ws://127.0.0.1:8000/ws"


class Parsing:
    """Command line arguments parser."""

    _parser: ArgumentsParameter

    def __init__(self) -> None:
        """Declare the command line arguments."""
        self._parser = argparse.ArgumentParser(prog="sensai-tui", description="Terminal client of a Sensai server.")
        self._parser.add_argument(
            "--url", default=DEFAULT_URL, help=f"websocket of the server (default: {DEFAULT_URL})"
        )
        self._parser.add_argument("--model", required=True, help="model to log in with")

    def parse_args(self, argv: Sequence[str] | None = None) -> argparse.Namespace:
        """Parse the command line arguments."""
        return self._parser.parse_args(argv)
