from pydantic import BaseModel, ConfigDict


class ProtocolModel(BaseModel):
    """Base of every protocol message: immutable, and rejecting any field it does not declare."""

    model_config = ConfigDict(frozen=True, extra="forbid")
