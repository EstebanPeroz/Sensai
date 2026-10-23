from sensai.error import SensaiError


class RagError(SensaiError):
    """Base error of the RAG."""

    def __init__(self, message: str) -> None:
        """Store `message` prefixed with the error kind."""
        super().__init__(message)
        self.message = "[RagError]" + message


class VectorStoreError(RagError):
    """Raised when chunks cannot be stored or read back."""

    def __init__(self, message: str) -> None:
        """Store `message` prefixed with the error kind."""
        super().__init__(message)
        self.message = "[VectorStoreError]" + message


class SourceError(RagError):
    """Base error of the sources a document is collected from."""

    def __init__(self, message: str) -> None:
        """Store `message` prefixed with the error kind."""
        super().__init__(message)
        self.message = "[SourceError]" + message


class SourceNotFoundError(SourceError):
    """Raised when a source is not a URL and no such file exists."""

    def __init__(self, message: str) -> None:
        """Store `message` prefixed with the error kind."""
        super().__init__(message)
        self.message = "[SourceNotFoundError]" + message


class UnsupportedFormatError(SourceError):
    """Raised when a source has an extension the collector does not handle."""

    def __init__(self, message: str) -> None:
        """Store `message` prefixed with the error kind."""
        super().__init__(message)
        self.message = "[UnsupportedFormatError]" + message


class SourceFetchError(SourceError):
    """Raised when a source exists but cannot be fetched or decoded."""

    def __init__(self, message: str) -> None:
        """Store `message` prefixed with the error kind."""
        super().__init__(message)
        self.message = "[SourceFetchError]" + message


class EmbeddingError(RagError):
    """Raised when the provider does not return the expected embeddings."""

    def __init__(self, message: str) -> None:
        """Store `message` prefixed with the error kind."""
        super().__init__(message)
        self.message = "[EmbeddingError]" + message
