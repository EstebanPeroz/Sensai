from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sensai.tools.tool import Tool


def build_tools() -> list[Tool]:
    """Build every tool the model may be given. Single entry point for the composition root.

    Concrete tools are imported inside this function, so a tool module can import
    `sensai.tools` without a circular import.
    """
    return []
