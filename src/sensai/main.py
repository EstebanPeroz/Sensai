from __future__ import annotations

import argparse
import sys
import threading
from pathlib import Path
from typing import TYPE_CHECKING

from sensai.config.settings import Settings, load_settings
from sensai.core.core import Core
from sensai.error import SensaiError
from sensai.ui.tui.core import UITextualAdapter

if TYPE_CHECKING:
    from collections.abc import Sequence

    from sensai.ui.adapter import UIAdapter


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


def launch_core(ui: UIAdapter, settings: Settings) -> None:
    """Build and run the Core loop, driven by `ui`. Meant to run off the main thread."""
    core = Core(ui, settings)
    core.run()


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point: run the program."""
    ui: UIAdapter = UITextualAdapter()
    args = parse_args(argv)
    try:
        settings = load_settings(args.config)
    except SensaiError as err:
        print(f"{err}", file=sys.stderr)
        return 1
    driver = threading.Thread(target=launch_core, args=(ui, settings), daemon=True)
    driver.start()
    ui.run()
    return 0


if __name__ == "__main__":
    main()
