from sensai.error import SensaiError


class DuplicateToolError(SensaiError):
    """Raised when several tools are registered under the same name."""

    def __init__(self, name: str) -> None:
        """Build the error from the duplicated tool `name`."""
        super().__init__(f"[DuplicateToolError] two tools are registered as '{name}'")
