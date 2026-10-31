from sensai_protocol.error import SensaiError


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
    """Raised when the content of a persona file does not describe a persona."""

    def __init__(self, path: str, reason: str) -> None:
        """Build the error from the `path` of the file (on the client, only shown) and the `reason` it was rejected."""
        self.path = path
        self.reason = reason
        super().__init__(f"invalid persona file '{path}': {reason}")


class DuplicatePersonaError(PersonaError):
    """Raised when a persona is given a name another persona already uses."""

    def __init__(self, persona_name: str) -> None:
        """Build the error from the `persona_name` that is already taken."""
        self.persona_name = persona_name
        super().__init__(f"a persona named '{persona_name}' already exists")
