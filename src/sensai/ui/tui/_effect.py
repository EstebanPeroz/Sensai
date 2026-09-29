def red(string: str) -> str:
    """Wrap `string` in Textual markup so it renders in red (used for errors)."""
    return _transform(string, "red")


def grey(string: str) -> str:
    """Wrap `string` in Textual markup so it renders in grey (used for thinking text)."""
    return _transform(string, "grey")


def bold(string: str) -> str:
    """Wrap `string` in Textual markup so it renders in bold (used for role labels)."""
    return _transform(string, "bold")


def _transform(string: str, version: str) -> str:
    """Wrap `string` in a Textual markup tag named `version` (e.g. "[red]...[/red]")."""
    return "[" + version + "]" + string + "[/" + version + "]"
