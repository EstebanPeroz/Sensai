from __future__ import annotations

import tomllib
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar

from sensai_protocol.server_messages import Role

from sensai_server.llm.persona.error import InvalidPersonaError, PersonaFileError
from sensai_server.memory.message import Message

if TYPE_CHECKING:
    from datetime import datetime
    from uuid import UUID


DEFAULT_PERSONA_FILE = "personas/analyst.toml"
MAX_PROMPT_LENGTH = 20_000  # characters; the prompt is sent with every message


@dataclass(frozen=True)
class Persona:
    """A reusable behaviour profile for the model: how it should answer, and what it is for."""

    # The fields a persona is authored with: the three keys of a persona file, and the
    # three that must hold text. The database assigns the id and the creation time.
    REQUIRED_FIELDS: ClassVar[tuple[str, ...]] = ("name", "description", "prompt")

    uuid: UUID | None
    name: str
    description: str
    prompt: str
    created_at: datetime | None = None

    def __post_init__(self) -> None:
        """Check the text fields, which come unchecked from the database or from a file."""
        invalid = [
            name
            for name in self.REQUIRED_FIELDS
            if not isinstance(getattr(self, name), str) or not getattr(self, name).strip()
        ]
        if invalid:
            raise InvalidPersonaError(invalid)

    @classmethod
    def from_toml(cls, text: str, source: str) -> Persona:
        """Parse the TOML `text` of a persona file into an unsaved persona.

        `source` is the path of the file (on the client), only shown in error messages.
        """
        try:
            data = tomllib.loads(text)
        except tomllib.TOMLDecodeError as err:
            raise PersonaFileError(source, f"invalid TOML ({err})") from err
        cls._check_keys(source, data)
        if len(data["prompt"]) > MAX_PROMPT_LENGTH:
            raise PersonaFileError(source, f"prompt longer than {MAX_PROMPT_LENGTH} characters")

        try:
            return cls(None, data["name"], data["description"], data["prompt"])
        except InvalidPersonaError as err:
            reason = "empty key(s): " + ", ".join(err.invalid_fields)
            raise PersonaFileError(source, reason) from err

    def show(self) -> dict[str, str]:
        """Return the fields identifying the persona, for display."""
        return {"name": self.name, "description": self.description}

    def to_message(self) -> Message:
        """Return the persona's prompt as the system message put before the conversation."""
        return Message(uuid=None, role=Role.SYSTEM, content=self.prompt)

    @classmethod
    def _check_keys(cls, source: str, data: dict[str, Any]) -> None:
        """Check that `data` holds every persona key, and nothing else, as text."""
        missing = [key for key in cls.REQUIRED_FIELDS if key not in data]
        unknown = sorted(key for key in data if key not in cls.REQUIRED_FIELDS)
        problems = []
        if missing:
            problems.append("missing key(s): " + ", ".join(missing))
        if unknown:
            problems.append("unknown key(s): " + ", ".join(unknown))
        if problems:
            raise PersonaFileError(source, "; ".join(problems))
        not_text = [key for key in cls.REQUIRED_FIELDS if not isinstance(data[key], str)]
        if not_text:
            reason = "key(s) that must be text: " + ", ".join(not_text)
            raise PersonaFileError(source, reason)
