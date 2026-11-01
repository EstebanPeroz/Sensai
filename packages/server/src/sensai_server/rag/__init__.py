"""The RAG of the conversations: collecting, chunking, storing and searching their documents."""

from sensai_server.rag.error import RagError
from sensai_server.rag.rag import Rag

__all__ = ["Rag", "RagError"]
