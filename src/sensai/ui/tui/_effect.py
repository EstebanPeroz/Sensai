def red(string: str) -> str:
    """TMP."""
    return _transform(string, "red")


def bold(string: str) -> str:
    """TMP."""
    return _transform(string, "bold")


def _transform(string: str, version: str) -> str:
    return "[" + version + "]" + string + "[/" + version + "]"
