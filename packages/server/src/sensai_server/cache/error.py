from sensai_protocol.error import SensaiError


class CacheError(SensaiError):
    """Raised when the semantic cache cannot reach Redis, or Redis refuses a command."""

    def __init__(self, message: str) -> None:
        """Build the error from `message`, prefixed with the error kind."""
        super().__init__(f"[CacheError] {message}")
