from sensai.error import SensaiError


class DatabaseError(SensaiError):
    """Tmp."""

    def __init__(self, message: str) -> None:
        """Tmp."""
        super().__init__(message)
        self.message = "[DatabaseError]" + message


class DbConnectionError(DatabaseError):
    """Tmp."""

    def __init__(self, message: str) -> None:
        """Tmp."""
        super().__init__(message)
        self.message = "[ConnectionError]" + message


class InvalidInstanceError(DatabaseError):
    """Tmp."""

    def __init__(self, message: str) -> None:
        """Tmp."""
        super().__init__(message)
        self.message = "[InvalidInstanceError]" + message
