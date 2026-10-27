from __future__ import annotations

from typing import TYPE_CHECKING

import uvicorn

from sensai_server.app import create_app
from sensai_server.parsing import Parsing

if TYPE_CHECKING:
    from collections.abc import Sequence


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point: parse the arguments and serve the app until stopped."""
    args = Parsing().parse_args(argv)
    uvicorn.run(create_app(), host=args.host, port=args.port)
    return 0
