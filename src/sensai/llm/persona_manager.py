from sensai.llm.persona import Persona, PersonaNotFoundError


class PersonaManager:
    """Manages loading and switching active personas."""

    def __init__(self, registry: dict[str, str] | None = None) -> None:
        """Inititate the persona manager class."""
        source = registry if registry is not None else {"analyst": "config/personas/analyst.toml"}
        self._registry = {name.lower(): path for name, path in source.items()}
        self._active_persona: Persona | None = None

    def register(self, name: str, file_path: str) -> None:
        """Register a new persona in the dictionnary."""
        self._registry[name.lower()] = file_path

    def get_registered_personas(self) -> dict[str, str]:
        """Return the personas dictionnary."""
        return self._registry

    def activate(self, name: str) -> Persona:
        """Activate a new persona."""
        key = name.lower()
        try:
            path = self._registry[key]
        except KeyError:
            raise PersonaNotFoundError(name, list(self._registry)) from None
        persona = Persona(key, path)
        self._active_persona = persona
        return persona
