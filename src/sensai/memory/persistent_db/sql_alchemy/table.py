from __future__ import annotations

from datetime import datetime
from uuid import UUID, uuid4

from sqlalchemy import JSON, DateTime, ForeignKey
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

from sensai.llm.message import Role


class Base(DeclarativeBase):
    """Base."""


class ToolCallTable(Base):
    """Base."""

    __tablename__ = "tool_call"

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    function_name: Mapped[str] = mapped_column()
    arguments: Mapped[dict] = mapped_column(JSON)

    message_id: Mapped[UUID] = mapped_column(
        ForeignKey("message.id", ondelete="CASCADE"),
        unique=True,
    )
    message: Mapped[MessageTable] = relationship(back_populates="tool_call")


class MessageTable(Base):
    """Base."""

    __tablename__ = "message"

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    role: Mapped[Role] = mapped_column()
    content: Mapped[str] = mapped_column()

    branch_id: Mapped[UUID] = mapped_column(ForeignKey("branch.id"), index=True)
    branch: Mapped[BranchTable] = relationship(back_populates="messages")

    tool_call: Mapped[ToolCallTable | None] = relationship(
        back_populates="message",
        uselist=False,
        cascade="all, delete-orphan",
        passive_deletes=True,
    )


class PersonaTable(Base):
    """Base."""

    __tablename__ = "persona"

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    name: Mapped[str] = mapped_column(unique=True)

    description: Mapped[str] = mapped_column()
    prompt: Mapped[str] = mapped_column()

    branch_id: Mapped[UUID] = mapped_column(ForeignKey("branch.id"), index=True)
    branch: Mapped[BranchTable] = relationship(back_populates="messages")


class BranchTable(Base):
    """Base."""

    __tablename__ = "branch"

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    name: Mapped[str] = mapped_column(unique=True)
    model_name: Mapped[str] = mapped_column(index=True)
    persona_id: Mapped[UUID | None] = mapped_column(ForeignKey("persona.id"))

    messages: Mapped[list[MessageTable]] = relationship(
        back_populates="branch",
        order_by="Message.created_at",
        cascade="all, delete-orphan",
    )
