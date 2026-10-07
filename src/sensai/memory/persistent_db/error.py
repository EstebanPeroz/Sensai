from sensai.error import SensaiError


class DbConnectionError(SensaiError):
    """Tmp."""

    def __init__(self, message: str) -> None:
        """Tmp."""
        super().__init__(message)
        self.message = "[DatabaseConnection]" + message
