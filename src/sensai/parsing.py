from __future__ import annotations

import argparse
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence


class Parsing:
    """Command line arguments parser."""

    def __init__(self) -> None:
        """Declare the command line arguments."""
        self._parser = argparse.ArgumentParser(prog="sensai", description="Local chatbot backed by Ollama.")
        self._parser.add_argument(
            "--config",
            type=Path,
            metavar="PATH",
            help="settings file to use instead of settings.toml",
        )
        self._parser.add_argument("--model", help="model to use instead of the provider default")

    def parse_args(self, argv: Sequence[str] | None = None) -> argparse.Namespace:
        """Parse the command line arguments."""
        return self._parser.parse_args(argv)
