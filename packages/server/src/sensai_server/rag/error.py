from sensai_protocol.error import SensaiError


class RagError(SensaiError):
    """Base error of the RAG. Each subclass only names its `kind`, shown before the message."""

    kind = "RagError"

    def __init__(self, message: str) -> None:
        """Build the error from `message`, prefixed with the error kind."""
        super().__init__(f"[{self.kind}] {message}")


class VectorStoreError(RagError):
    """Raised when chunks cannot be stored or read back."""

    kind = "VectorStoreError"


class SourceError(RagError):
    """Base error of the sources a document is collected from."""

    kind = "SourceError"


class SourceNotFoundError(SourceError):
    """Raised when a source does not exist."""

    kind = "SourceNotFoundError"


class UnsupportedFormatError(SourceError):
    """Raised when a source has an extension the collector does not handle."""

    kind = "UnsupportedFormatError"


class SourceFetchError(SourceError):
    """Raised when a source exists but cannot be fetched or decoded."""

    kind = "SourceFetchError"
