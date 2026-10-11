from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

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


class PersonaFileError(PersonaError):
    """Raised when a persona file cannot be read or does not describe a persona."""

    def __init__(self, path: Path, reason: str) -> None:
        """Build the error from the `path` of the file and the `reason` it was rejected."""
        self.path = path
        self.reason = reason
        super().__init__(f"invalid persona file '{path}': {reason}")


class DuplicatePersonaError(PersonaError):
    """Raised when a persona is given a name another persona already uses."""

    def __init__(self, persona_name: str) -> None:
        """Build the error from the `persona_name` that is already taken."""
        self.persona_name = persona_name
        super().__init__(f"a persona named '{persona_name}' already exists")


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
    def from_file(cls, path: str | Path) -> Persona:
        """Read the persona file at `path` and return the persona it describes."""
        source = Path(path)
        data = cls._read_file(source)
        cls._check_keys(source, data)

        try:
            return cls(None, data["name"], data["description"], data["prompt"])
        except InvalidPersonaError as err:
            reason = "empty key(s): " + ", ".join(err.invalid_fields)
            raise PersonaFileError(source, reason) from err

    def show(self) -> dict[str, str]:
        """Return the fields identifying the persona, for display."""
        return {"name": self.name, "description": self.description}

    @staticmethod
    def _read_file(path: Path) -> dict[str, Any]:
        """Load the TOML file at `path`, reporting why it could not be read."""
        try:
            with path.open("rb") as file:
                return tomllib.load(file)
        except FileNotFoundError:
            reason = "file not found"
        except IsADirectoryError:
            reason = "path is a directory"
        except OSError as err:
            reason = f"cannot read file ({err.strerror})"
        except UnicodeDecodeError:
            reason = "file is not valid UTF-8"
        except tomllib.TOMLDecodeError as err:
            reason = f"invalid TOML ({err})"
        raise PersonaFileError(path, reason)

    @classmethod
    def _check_keys(cls, path: Path, data: dict[str, Any]) -> None:
        """Check that `data` holds every persona key, and nothing else, as text."""
        missing = [key for key in cls.REQUIRED_FIELDS if key not in data]
        unknown = sorted(key for key in data if key not in cls.REQUIRED_FIELDS)
        problems = []
        if missing:
            problems.append("missing key(s): " + ", ".join(missing))
        if unknown:
            problems.append("unknown key(s): " + ", ".join(unknown))
        if problems:
            raise PersonaFileError(path, "; ".join(problems))
        not_text = [key for key in cls.REQUIRED_FIELDS if not isinstance(data[key], str)]
        if not_text:
            reason = "key(s) that must be text: " + ", ".join(not_text)
            raise PersonaFileError(path, reason)

    def apply_to_payload(self, messages: list[dict]) -> dict:
        """Prepare to the paylod, its content."""
        return {"messages": [{"role": "system", "content": self.prompt}, *messages]}


DEFAULT_PERSONA_FILE = "personas/analyst.toml"
