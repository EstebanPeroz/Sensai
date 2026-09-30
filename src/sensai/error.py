class SensaiError(Exception):
    """Tmp."""

    message: str

    def __init__(self, message: str) -> None:
        """Tmp."""
        super().__init__(message)
        self.message = "[Error]" + message
