from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

from sensai.error import SensaiError

if TYPE_CHECKING:
    from datetime import datetime
    from uuid import UUID


class PersonaError(SensaiError):
    """Base class for all persona-related errors."""

    def __init__(self, message: str) -> None:
        """Build the error, tagged with the name of the raised class, from `message`."""
        super().__init__(f"[{type(self).__name__}] {message}")


class InvalidPersonaError(PersonaError):
    """Raised when a persona is built from missing or empty fields."""

    def __init__(self, invalid_fields: list[str]) -> None:
        """Build the error from the `invalid_fields` that were missing or empty."""
        self.invalid_fields = invalid_fields
        names = ", ".join(invalid_fields) or "none"
        super().__init__("missing or empty field(s): " + names)


class PersonaNotRegisteredError(PersonaError):
    """Raised when no stored persona goes by a given name."""

    def __init__(self, persona_name: str, available: list[str]) -> None:
        """Build the error from the unknown `persona_name` and the names that are `available`."""
        self.persona_name = persona_name
        self.available = available
        names = ", ".join(sorted(available)) or "none"
        super().__init__(f"persona '{persona_name}' is not registered, available: {names}")


class DuplicatePersonaError(PersonaError):
    """Raised when a persona is given a name another persona already uses."""

    def __init__(self, persona_name: str) -> None:
        """Build the error from the `persona_name` that is already taken."""
        self.persona_name = persona_name
        super().__init__(f"a persona named '{persona_name}' already exists")


@dataclass(frozen=True)
class Persona:
    """A reusable behaviour profile for the model: how it should answer, and what it is for.

    Personas are plain data, stored in the `persona` table and never read from a file.
    `uuid` and `created_at` are the fields of the stored record, so they are None for a
    persona that has not been stored yet, exactly like `Message.uuid`.
    """

    REQUIRED_FIELDS: ClassVar[tuple[str, ...]] = ("name", "description", "prompt")

    uuid: UUID | None
    name: str
    description: str
    prompt: str
    created_at: datetime | None = None

    def __post_init__(self) -> None:
        """Check the text fields, which come unchecked from the database or from the caller."""
        invalid = [
            name
            for name in self.REQUIRED_FIELDS
            if not isinstance(getattr(self, name), str) or not getattr(self, name).strip()
        ]
        if invalid:
            raise InvalidPersonaError(invalid)

    def show(self) -> dict[str, str]:
        """Return the fields identifying the persona, for display."""
        return {"name": self.name, "description": self.description}

    def apply_to_payload(self, messages: list[dict]) -> dict:
        """Return a chat payload fragment where this persona's prompt comes before `messages`."""
        return {"messages": [{"role": "system", "content": self.prompt}, *messages]}


# The persona a fresh database is seeded with, so the assistant behaves sensibly
# before anyone creates one (US-24). It replaces config/personas/analyst.toml.
DEFAULT_PERSONA = Persona(
    uuid=None,
    name="Analyst",
    description="An analytical assistant that evaluates inputs through structured decomposition "
    "and objective evaluation.",
    prompt="""You are an expert analyst. When presented with information or a problem:
1. Deconstruct the context and identify key facts, assumptions, and variables.
2. Evaluate underlying patterns, logical dependencies, and potential edge cases.
3. Provide a clear, objective analysis followed by actionable conclusions or recommendations.""",
)
