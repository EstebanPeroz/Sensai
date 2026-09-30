from __future__ import annotations

import argparse
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the command line arguments."""
    parser = argparse.ArgumentParser(prog="sensai", description="Local chatbot backed by Ollama.")
    parser.add_argument(
        "--config",
        type=Path,
        metavar="PATH",
        help="settings file to use instead of settings.toml",
    )
    parser.add_argument("--model", help="model to use instead of the provider default")
    return parser.parse_args(argv)
