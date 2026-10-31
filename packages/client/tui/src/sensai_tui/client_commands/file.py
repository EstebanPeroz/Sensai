from __future__ import annotations

import asyncio
from pathlib import Path

from sensai_protocol import MAX_FILE_SIZE, FileContent, ReadFileParams


class ClientCommandFailedError(Exception):
    """Raised by a client command that could not run, with the `reason` sent back to the server."""

    def __init__(self, reason: str) -> None:
        """Build the error from the `reason` the command failed."""
        super().__init__(reason)
        self.reason = reason


async def read_file(params: ReadFileParams) -> FileContent:
    """Read the file at `params.path`, relative to the client's current directory, as UTF-8 text.

    The content is not parsed: the server does it. Raises ClientCommandFailedError when the file cannot be read.
    """
    content = await asyncio.to_thread(_read_text, params.path)
    return FileContent(content=content)


def _read_text(raw_path: str) -> str:
    """Return the text of the file at `raw_path`, checking first that it is a file small enough to be sent."""
    path = Path(raw_path).expanduser()
    try:
        reason = _unreadable_reason(path)
        if reason is not None:
            raise ClientCommandFailedError(reason)
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as err:
        reason = "not valid UTF-8"
        raise ClientCommandFailedError(reason) from err
    except OSError as err:
        reason = f"cannot read file ({err.strerror or err})"
        raise ClientCommandFailedError(reason) from err
    if len(text) > MAX_FILE_SIZE:
        reason = f"file too large (more than {MAX_FILE_SIZE} characters)"
        raise ClientCommandFailedError(reason)
    return text


def _unreadable_reason(path: Path) -> str | None:
    """Return why the file at `path` cannot be sent, checked before reading it, or None if it can be read."""
    if not path.exists():
        return "file not found"
    if path.is_dir():
        return "path is a directory"
    if path.stat().st_size > MAX_FILE_SIZE:
        return f"file too large (more than {MAX_FILE_SIZE} bytes)"
    return None
