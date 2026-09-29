from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from sensai.config.settings import ConfigError, Settings, load_settings

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
    parser.add_argument("--model", help="model to use, overrides llm.model from the settings file")
    return parser.parse_args(argv)


def build_settings(args: argparse.Namespace) -> Settings:
    """Load the settings file and apply the command line overrides on top of it."""
    settings = load_settings(args.config)
    if args.model is not None:
        settings.llm.model = args.model
    return settings


def main(argv: Sequence[str] | None = None) -> int:
    """Run the program."""
    try:
        build_settings(parse_args(argv))
    except ConfigError as err:
        print(f"sensai: error: {err}", file=sys.stderr)
        return 2
    return 0
