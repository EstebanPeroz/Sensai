#!/usr/bin/env python3

from __future__ import annotations

import threading
from typing import TYPE_CHECKING

from sensai.core.core import Core
from sensai.ui.tui.core import UITextualAdapter

if TYPE_CHECKING:
    from sensai.ui.adapter import UIAdapter


def launch_core(ui: UIAdapter) -> None:
    """Build and run the Core loop, driven by `ui`. Meant to run off the main thread."""
    core = Core(ui)
    core.run()


def main() -> None:
    """Entry point: run the program."""
    ui: UIAdapter = UITextualAdapter()

    driver = threading.Thread(target=launch_core, args=(ui,), daemon=True)
    driver.start()
    ui.run()


if __name__ == "__main__":
    main()
