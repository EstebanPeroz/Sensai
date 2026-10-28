from textual.markup import escape

USER_PROMPT_BACKGROUND: str = "grey"


TEXTURAL_CSS = f"""
AppHeader {{
    height: 5%;
    width: 100%;
}}
AppHeader #header-brand {{
    width: 1fr;
    height: 100%;
    content-align: left middle;
}}
AppHeader #header-right {{
    width: 1fr;
    height: 100%;
    align: right middle;
}}
AppHeader #header-model {{
    width: auto;
    height: auto;
    margin-right: 2;
}}
AppHeader #header-cwd {{
    width: auto;
    height: auto;
}}
ListView#chat {{
    width: 100%;
}}
ListView#chat > ListItem {{
    width: 100%;
}}
ListView#chat > ListItem HorizontalGroup {{
    width: 100%;
    height: auto;
}}
ListView#chat > ListItem Label {{
    width: 100%;
    height: auto;
}}
ListView#chat > ListItem.user-message Label {{
    background: {USER_PROMPT_BACKGROUND};
}}
ListView#chat > ListItem.user-message {{
    margin-top: 1;
}}
ListView#chat > ListItem.user-message:first-child {{
    margin-top: 0;
}}
CompletionList {{
    display: none;
    height: auto;
    max-height: 8;
}}
Input#input {{
    height: 3;
    border: round pink;
    padding: 0 1;
}}
"""


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


def italic(string: str) -> str:
    """Wrap `string` in Textual markup so it renders in italic (used for commands and system messages)."""
    return _transform(string, "italic")


def background(string: str, color: str) -> str:
    """Wrap `string` in Textual markup so it renders with `color` as its background."""
    return _transform(string, "on " + color)


def _transform(string: str, version: str) -> str:
    """Wrap `string` in a Textual markup tag named `version` (e.g. "[red]...[/red]")."""
    return "[" + version + "]" + string + "[/" + version + "]"


def format_message(mode: str, text: str) -> str:
    """Render `text` with the markup appropriate to its rendering `mode`."""
    text = escape(text)
    match mode:
        case "message":
            formatted = "> " + text
        case "command":
            formatted = "> " + italic(text)
        case "system":
            formatted = italic(bold("* system:")) + "\n" + text
        case "thinking":
            formatted = grey("• " + text)
        case "content":
            formatted = white("• " + text)
        case "error":
            formatted = red("@ " + text)
        case _:
            formatted = text
    return formatted
