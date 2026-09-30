from sensai.error import SensaiError


class RequestCallError(SensaiError):
    """Tmp."""

    def __init__(self, message: str) -> None:
        """Tmp."""
        super().__init__("[RequestCallError]" + message)


class RequestStatusError(SensaiError):
    """Tmp."""

    def __init__(self, code: int) -> None:
        """Tmp."""
        super().__init__("[RequestStatusError] status " + str(code))


class JsonError(SensaiError):
    """Tmp."""

    def __init__(self, message: str = "") -> None:
        """Tmp."""
        super().__init__("[JsonError]" + message)
