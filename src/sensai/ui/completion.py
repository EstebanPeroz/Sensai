from collections.abc import Callable

type CompletionSource = Callable[[list[str]], list[str]]
"""Gives the values a command's next word can take, from the words already typed after the command."""


class Completer:
    """Completes the input line, asking a command for its values only when a new word starts."""

    _sources: dict[str, CompletionSource]
    _values: list[str]

    def __init__(self, sources: dict[str, CompletionSource] | None = None) -> None:
        """Start with the `sources` of each command, by name."""
        self._sources = sources or {}
        self._fetched: tuple[str, tuple[str, ...]] | None = None
        self._values: list[str] = []

    def set_sources(self, sources: dict[str, CompletionSource]) -> None:
        """Replace the sources and forget the values fetched from the previous ones."""
        self._sources = sources
        self._fetched = None
        self._values = []

    def complete(self, text: str) -> list[str]:
        """Return the full input lines that complete `text`.

        Without a space, `text` is completed as a command name. After one, the command's source is called with the
        finished words, only when they differ from the last call (a word was started or removed); the typing of
        the current word just filters the values already fetched.
        """
        if not text.startswith("/"):
            return []
        words = text.split()
        if not words:
            return []
        command, *args = words
        trailing_space = text[-1].isspace()
        if not args and not trailing_space:
            return [name for name in self._sources if name.startswith(command)]
        source: CompletionSource | None = self._sources.get(command)
        if source is None:
            return []
        if trailing_space:
            finished, partial = args, ""
        else:
            *finished, partial = args
        key = (command, tuple(finished))
        if key != self._fetched:
            self._values = source(finished)
            self._fetched = key
        prefix = " ".join([command, *finished])
        return [f"{prefix} {value}" for value in self._values if value.startswith(partial)]
