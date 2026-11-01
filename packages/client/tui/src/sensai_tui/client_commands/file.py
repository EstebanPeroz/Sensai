from __future__ import annotations

import asyncio
import base64
from pathlib import Path

from sensai_protocol import (
    MAX_BINARY_FILE_SIZE,
    MAX_FILE_SIZE,
    BinaryFileContent,
    FileContent,
    FileWritten,
    ReadFileParams,
    WriteFileParams,
)


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


async def read_binary_file(params: ReadFileParams) -> BinaryFileContent:
    """Read the file at `params.path`, relative to the client's current directory, as bytes encoded in base64.

    The content is not parsed: the server does it. Raises ClientCommandFailedError when the file cannot be read.
    """
    data = await asyncio.to_thread(_read_bytes, params.path)
    return BinaryFileContent(content=base64.b64encode(data).decode("ascii"))


async def write_file(params: WriteFileParams) -> FileWritten:
    """Write `params.content` to the file at `params.path`, relative to the client's current directory, as UTF-8.

    Raises ClientCommandFailedError when the file cannot be written.
    """
    path = await asyncio.to_thread(_write_text, params.path, params.content)
    return FileWritten(path=str(path))


def _read_text(raw_path: str) -> str:
    """Return the text of the file at `raw_path`, checking first that it is a file small enough to be sent."""
    path = Path(raw_path).expanduser()
    try:
        reason = _unreadable_reason(path, MAX_FILE_SIZE)
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


def _read_bytes(raw_path: str) -> bytes:
    """Return the bytes of the file at `raw_path`, checking first that it is a file small enough to be sent."""
    path = Path(raw_path).expanduser()
    try:
        reason = _unreadable_reason(path, MAX_BINARY_FILE_SIZE)
        if reason is not None:
            raise ClientCommandFailedError(reason)
        data = path.read_bytes()
    except OSError as err:
        reason = f"cannot read file ({err.strerror or err})"
        raise ClientCommandFailedError(reason) from err
    if len(data) > MAX_BINARY_FILE_SIZE:
        reason = f"file too large (more than {MAX_BINARY_FILE_SIZE} bytes)"
        raise ClientCommandFailedError(reason)
    return data


def _unreadable_reason(path: Path, max_size: int) -> str | None:
    """Return why the file at `path` cannot be sent, checked before reading it, or None if it can be read.

    `max_size` is the most bytes the file may hold.
    """
    if not path.exists():
        return "file not found"
    if path.is_dir():
        return "path is a directory"
    if path.stat().st_size > max_size:
        return f"file too large (more than {max_size} bytes)"
    return None


def _write_text(raw_path: str, content: str) -> Path:
    """Write `content` to the file at `raw_path`, replacing it if it exists, and return its absolute path."""
    path = Path(raw_path).expanduser()
    try:
        reason = _unwritable_reason(path)
        if reason is not None:
            raise ClientCommandFailedError(reason)
        path.write_text(content, encoding="utf-8")
    except OSError as err:
        reason = f"cannot write file ({err.strerror or err})"
        raise ClientCommandFailedError(reason) from err
    return path.resolve()


def _unwritable_reason(path: Path) -> str | None:
    """Return why the file at `path` cannot be written, checked before writing it, or None if it can be written."""
    if path.is_dir():
        return "path is a directory"
    if not path.parent.is_dir():
        return "directory not found"
    return None
