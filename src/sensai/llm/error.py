from sensai.error import SensaiError


class RequestCallError(SensaiError):
    """Raised when an HTTP request to a provider could not be sent (e.g. connection refused, timeout)."""

    def __init__(self, message: str) -> None:
        """Build the error from the failing call and the underlying exception's message."""
        super().__init__("[RequestCallError]" + message)


class RequestStatusError(SensaiError):
    """Raised when a provider responds with a non-success HTTP status code."""

    def __init__(self, code: int) -> None:
        """Build the error from the response's status `code`."""
        super().__init__("[RequestStatusError] status " + str(code))


class JsonError(SensaiError):
    """Raised when a provider's response body could not be decoded as JSON."""

    def __init__(self, message: str = "") -> None:
        """Build the error, optionally with additional context in `message`."""
        super().__init__("[JsonError]" + message)


class ModelError(SensaiError):
    """Base class for exceptions related to model operations."""
