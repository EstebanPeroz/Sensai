import tomllib
from pathlib import Path


class PersonaError(Exception):
    """Base class for all persona-related errors."""


class PersonaNotFoundError(PersonaError):
    """Raised when a persona name is not in the registry."""

    def __init__(self, persona_name: str, available: list[str]) -> None:
        """Init persona not found error class."""
        self.persona_name = persona_name
        self.available = available
        names = ", ".join(sorted(available)) or "none"
        super().__init__(f"Persona '{persona_name}' not found in registry. Available: {names}.")


class PersonaConfigError(PersonaError):
    """Raised when a persona config file is missing, unreadable or invalid."""

    def __init__(self, path: Path, reason: str) -> None:
        """Init personconfig error class."""
        self.path = path
        self.reason = reason
        super().__init__(f"Invalid persona config '{path}': {reason}")


class Persona:
    """A persona loaded from a TOML config file."""

    REQUIRED_FIELDS = ("description", "system_prompt", "version")

    def __init__(self, registry_name: str, file_path: str | Path) -> None:
        """Init the persona class."""
        self.registry_name = registry_name
        self.file_path = Path(file_path)
        data = self._load_file()
        self._validate(data)
        self.name = data.get("name", registry_name)
        self.version = str(data.get("version", "1.0"))
        self.description = data["description"]
        self.system_prompt = data["system_prompt"]

    def _load_file(self) -> dict:
        """Load a file for a new persona."""
        try:
            with self.file_path.open("rb") as f:
                return tomllib.load(f)
        except FileNotFoundError as err:
            raise PersonaConfigError(self.file_path, "file not found") from err
        except OSError as err:
            raise PersonaConfigError(self.file_path, f"cannot read file ({err.strerror})") from err
        except tomllib.TOMLDecodeError as err:
            raise PersonaConfigError(self.file_path, f"invalid TOML ({err})") from err

    def _validate(self, data: dict) -> None:
        """Check all necessaries values are here."""
        invalid = [
            field for field in self.REQUIRED_FIELDS if not isinstance(data.get(field), str) or not data[field].strip()
        ]
        if invalid:
            raise PersonaConfigError(self.file_path, f"missing or empty field(s): {', '.join(invalid)}")

    def show(self) -> list:
        """Show variables."""
        return [self.name, self.version, self.description]

    def apply_to_payload(self, messages: list[dict]) -> dict:
        """Prepare to the paylod, its content."""
        return {"messages": [{"role": "system", "content": self.system_prompt}, *messages]}
