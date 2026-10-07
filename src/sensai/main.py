from __future__ import annotations

import sys
import threading
from typing import TYPE_CHECKING

from sensai.config.settings import load_settings
from sensai.core.core import Core
from sensai.error import SensaiError
from sensai.parsing import Parsing
from sensai.ui.tui.core import UITextualAdapter

if TYPE_CHECKING:
    from collections.abc import Sequence

    from sensai.ui.adapter import UIAdapter


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point: run the program."""
    ui: UIAdapter = UITextualAdapter()
    args = Parsing().parse_args(argv)
    try:
        settings = load_settings(args.config)
        core = Core(ui, settings, args)
    except SensaiError as err:
        print(f"{err}", file=sys.stderr)
        return 1

    driver = threading.Thread(target=core.run, daemon=True)
    driver.start()
    ui.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
