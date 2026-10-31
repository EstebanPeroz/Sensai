from sensai_protocol.error import SensaiError


class DatabaseError(SensaiError):
    """Base error of the persistent database."""

    def __init__(self, message: str) -> None:
        """Store `message` prefixed with the error kind."""
        super().__init__(message)
        self.message = "[DatabaseError]" + message


class DbConnectionError(DatabaseError):
    """Raised when the connection to the database fails or is not configured."""

    def __init__(self, message: str) -> None:
        """Store `message` prefixed with the error kind."""
        super().__init__(message)
        self.message = "[ConnectionError]" + message


class InvalidInstanceError(DatabaseError):
    """Raised when an id does not match any stored record."""

    def __init__(self, message: str) -> None:
        """Store `message` prefixed with the error kind."""
        super().__init__(message)
        self.message = "[InvalidInstanceError]" + message
