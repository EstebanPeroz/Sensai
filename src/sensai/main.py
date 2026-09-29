from __future__ import annotations

import sys
from typing import TYPE_CHECKING

from sensai.config.settings import ConfigError, build_settings
from sensai.parsing import parse_args

if TYPE_CHECKING:
    from collections.abc import Sequence


def main(argv: Sequence[str] | None = None) -> int:
    """Run the program."""
    args = parse_args(argv)
    try:
        build_settings(args.config, model=args.model)
    except ConfigError as err:
        print(f"sensai: error: {err}", file=sys.stderr)
        return 2
    return 0
