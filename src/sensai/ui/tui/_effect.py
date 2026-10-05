def red(string: str) -> str:
    """Wrap `string` in Textual markup so it renders in red (used for errors)."""
    return _transform(string, "red")


def grey(string: str) -> str:
    """Wrap `string` in Textual markup so it renders in grey (used for thinking text)."""
    return _transform(string, "grey")


def white(string: str) -> str:
    """Wrap `string` in Textual markup so it renders in white (used for assistant content)."""
    return _transform(string, "white")


def bold(string: str) -> str:
    """Wrap `string` in Textual markup so it renders in bold (used for role labels)."""
    return _transform(string, "bold")


def background(string: str, color: str) -> str:
    """Wrap `string` in Textual markup so it renders with `color` as its background."""
    return _transform(string, "on " + color)


def _transform(string: str, version: str) -> str:
    """Wrap `string` in a Textual markup tag named `version` (e.g. "[red]...[/red]")."""
    return "[" + version + "]" + string + "[/" + version + "]"
