from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID, uuid4

from sqlalchemy import DateTime, ForeignKey
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

from sensai.llm.message import Role  # noqa: TC001


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
        """Tmp."""
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

    name: Mapped[str] = mapped_column(unique=True)
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
