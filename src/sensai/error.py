class SensaiError(Exception):
    """Base class for every error raised by Sensai's own code."""

    message: str

    def __init__(self, message: str) -> None:
        """Prefix `message` for display and pass it to the base exception."""
        super().__init__(message)
        self.message = "[Error]" + message
