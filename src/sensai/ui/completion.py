def complete(text: str, completions: dict[str, list[str]]) -> list[str]:
    """Return the full input lines that complete `text`.

    `completions` maps each command to the values its argument can take.
    Without a space, `text` is completed as a command name; after one, as that command's argument.
    """
    if not text.startswith("/"):
        return []
    command, sep, arg = text.partition(" ")
    if not sep:
        return [name for name in completions if name.startswith(command)]
    return [f"{command} {value}" for value in completions.get(command, []) if value.startswith(arg)]
