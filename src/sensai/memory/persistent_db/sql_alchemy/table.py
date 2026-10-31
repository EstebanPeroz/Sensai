from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID, uuid4

from sqlalchemy import JSON, DateTime, ForeignKey
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

from sensai.llm.message import Role  # noqa: TC001
from sensai.monitoring.interaction import InteractionStatus  # noqa: TC001


class Base(DeclarativeBase):
    """Base."""

    __abstract__ = True

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(tz=UTC),
    )


class MessageTable(Base):
    """Base."""

    __tablename__ = "message"

    role: Mapped[Role] = mapped_column()
    content: Mapped[str] = mapped_column()

    branch_id: Mapped[UUID] = mapped_column(ForeignKey("branch.id", ondelete="CASCADE"), index=True)
    branch: Mapped[BranchTable] = relationship(back_populates="messages")

    def copy(self) -> MessageTable:
        """Return an unsaved copy of the message, keeping its creation time."""
        new_message = MessageTable()
        new_message.created_at = self.created_at
        new_message.role = self.role
        new_message.content = self.content
        return new_message


class PersonaTable(Base):
    """Base."""

    __tablename__ = "persona"

    name: Mapped[str] = mapped_column(unique=True)

    description: Mapped[str] = mapped_column()
    prompt: Mapped[str] = mapped_column()

    branches: Mapped[list[BranchTable]] = relationship(back_populates="persona", passive_deletes=True)


class BranchTable(Base):
    """Base."""

    __tablename__ = "branch"

    name: Mapped[str] = mapped_column()
    model_name: Mapped[str] = mapped_column(index=True)
    persona_id: Mapped[UUID | None] = mapped_column(ForeignKey("persona.id", ondelete="SET NULL"))
    persona: Mapped[PersonaTable | None] = relationship(back_populates="branches")

    messages: Mapped[list[MessageTable]] = relationship(
        back_populates="branch",
        order_by="MessageTable.created_at",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )

    def copy(self, name: str) -> BranchTable:
        """Return an unsaved deep copy named `name`; ids are assigned on flush."""
        new_branch = BranchTable()
        new_branch.name = name
        new_branch.model_name = self.model_name
        new_branch.persona_id = self.persona_id
        for message in self.messages:
            new_branch.messages.append(message.copy())
        return new_branch


class SummaryTable(Base):
    """Summary of the oldest messages of a branch; removed with its branch."""

    __tablename__ = "summary"

    branch_id: Mapped[UUID] = mapped_column(ForeignKey("branch.id", ondelete="CASCADE"), unique=True)
    covered: Mapped[int] = mapped_column()
    content: Mapped[str] = mapped_column()


class InteractionTable(Base):
    """One logged call to the model; kept when its branch is removed."""

    __tablename__ = "interaction"

    created_at: Mapped[datetime] = mapped_column(  # the dashboard filters and sorts on it
        DateTime(timezone=True),
        default=lambda: datetime.now(tz=UTC),
        index=True,
    )
    model: Mapped[str] = mapped_column(index=True)
    status: Mapped[InteractionStatus] = mapped_column(index=True)
    latency_ms: Mapped[float] = mapped_column()
    branch_id: Mapped[UUID | None] = mapped_column(ForeignKey("branch.id", ondelete="SET NULL"), index=True)
    persona: Mapped[str | None] = mapped_column()
    request: Mapped[list[dict] | None] = mapped_column(JSON)
    response: Mapped[str | None] = mapped_column()
    thinking: Mapped[str | None] = mapped_column()
    tool_calls: Mapped[list[dict]] = mapped_column(JSON)
    error: Mapped[str | None] = mapped_column()
    first_token_ms: Mapped[float | None] = mapped_column()
    prompt_tokens: Mapped[int | None] = mapped_column()
    completion_tokens: Mapped[int | None] = mapped_column()
    total_duration_ms: Mapped[float | None] = mapped_column()
    load_duration_ms: Mapped[float | None] = mapped_column()
    prompt_eval_duration_ms: Mapped[float | None] = mapped_column()
    eval_duration_ms: Mapped[float | None] = mapped_column()
