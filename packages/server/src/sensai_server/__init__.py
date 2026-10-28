from __future__ import annotations

import sys
from typing import TYPE_CHECKING

import uvicorn

from sensai_server.app import create_app
from sensai_server.config.settings import ConfigError, load_settings
from sensai_server.parsing import Parsing

if TYPE_CHECKING:
    from collections.abc import Sequence


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point: parse the arguments and serve the app until stopped."""
    args = Parsing().parse_args(argv)
    try:
        settings = load_settings(args.config)
    except ConfigError as err:
        print(err.message, file=sys.stderr)
        return 1
    uvicorn.run(create_app(settings), host=args.host, port=args.port)
    return 0
