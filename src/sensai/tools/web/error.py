from sensai.error import SensaiError


class WebBackendError(SensaiError):
    """Raised when a web backend cannot answer (network, timeout, rate limit, bad response)."""

    backend: str
    reason: str

    def __init__(self, backend: str, reason: str) -> None:
        """Build the error from the failing backend's name and the `reason` it failed."""
        super().__init__(f"[WebBackendError] {backend}: {reason}")
        self.backend = backend
        self.reason = reason
