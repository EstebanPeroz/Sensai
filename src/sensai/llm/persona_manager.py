from sensai.llm.persona import Persona


class PersonaManager:
    """Manages loading, caching, and switching active personas."""

    def __init__(self, registry: dict[str, str] | None = None) -> None:
        """Initialize a persona manager."""
        self._registry = registry or {"analyst": "config/personas/analyst.toml"}
        self._active_persona: Persona | None = None

    def register(self, name: str, file_path: str) -> None:
        """Register a new persona."""
        self._registry[name.lower()] = file_path

    def set_active(self, name: str) -> Persona:
        """Active an other persona."""
        self._active_persona = Persona(name)
        return self._active_persona

    def active(self) -> Persona | None:
        """Launch the persona."""
        if not self._active_persona:
            return None
        return self._active_persona
