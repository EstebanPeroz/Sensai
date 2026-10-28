from collections.abc import Callable

type CompletionRequest = Callable[[str], None]
"""Asks the server for the values of the input line `text`; the answer comes back through `Completer.receive`."""


def _split(text: str) -> tuple[str, list[str], str] | None:
    """Split a command line into its command, finished words and partial word, or None if it has no arguments."""
    command, *args = text.split()
    if not args and not text[-1].isspace():
        return None
    if text[-1].isspace():
        return command, args, ""
    *finished, partial = args
    return command, finished, partial


class Completer:
    """Completes the input line, asking the server for a command's values only when a new word starts."""

    _commands: list[str]
    _values: list[str]

    def __init__(self, request: CompletionRequest | None = None, commands: list[str] | None = None) -> None:
        """Start with the command names, and the `request` callback asking the server for argument values."""
        self._request = request
        self._commands = commands or []
        self._fetched: tuple[str, tuple[str, ...]] | None = None
        self._values = []

    def set_commands(self, commands: list[str]) -> None:
        """Replace the command names and forget the values fetched for the previous ones."""
        self._commands = commands
        self._fetched = None
        self._values = []

    def complete(self, text: str) -> list[str]:
        """Return the full input lines that complete `text`.

        Without a space, `text` is completed as a command name. After one, the server is asked for the values of
        the finished words, only when they differ from the last request (a word was started or removed); the
        typing of the current word just filters the values already received.
        """
        if not text.startswith("/") or not text.split():
            return []
        parts = _split(text)
        if parts is None:
            return [name for name in self._commands if name.startswith(text)]
        command, finished, partial = parts
        if command not in self._commands:
            return []
        key = (command, tuple(finished))
        if key != self._fetched:
            self._fetched = key
            self._values = []
            if self._request is not None:
                self._request(text)
        prefix = " ".join([command, *finished])
        return [f"{prefix} {value}" for value in self._values if value.startswith(partial)]

    def receive(self, text: str, values: list[str]) -> bool:
        """Store the `values` answering `text`, unless the input moved to other words since. Return if stored."""
        if not text.startswith("/") or not text.split():
            return False
        parts = _split(text)
        if parts is None:
            return False
        command, finished, _ = parts
        if (command, tuple(finished)) != self._fetched:
            return False
        self._values = values
        return True
