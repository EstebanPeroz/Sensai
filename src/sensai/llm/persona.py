import tomllib
from pathlib import Path


class PersonaNotFoundError(ValueError):
    """Raised when a persona is not found in the registry."""

    def __init__(self, persona_name: str) -> None:
        """Initialize the exception with the persona name."""
        super().__init__(f"Persona '{persona_name}' not found in registry.")


class Persona:
    """Class used to load and create a persona from a config file."""

    def __init__(self, name: str) -> None:
        """Initialize a persona from its registry name."""
        self.registry_name = name.lower()
        self.file_path = "src.sensai.config." + name + ".toml"
        self._config_data = self._load_file()
        self.name = self.get_name_from_file()
        self.version = self.get_version_from_file()
        self.description = self.get_description_from_file()
        self.system_prompt = self.get_system_prompt_from_file()

    def _load_file(self) -> dict:
        """Help method to load the TOML file data into a dictionary."""
        file_path = Path(self.file_path)
        if not file_path.exists():
            raise PersonaNotFoundError(self.registry_name)
        with file_path.open("rb") as file:
            return tomllib.load(file)

    def get_version_from_file(self) -> str:
        """Retrieve the version from the loaded data."""
        return str(self._config_data.get("version", "1.0"))

    def get_system_prompt_from_file(self) -> str:
        """Retrieve the system prompt from the loaded data."""
        config_data = self._config_data.get("system_prompt", "")
        if not config_data:
            raise PersonaNotFoundError(self.registry_name)
        return config_data

    def get_description_from_file(self) -> str:
        """Retrieve the description from the loaded data."""
        description = self._config_data.get("description", "")
        if not description:
            raise PersonaNotFoundError(self.registry_name)
        return description

    def get_name_from_file(self) -> str:
        """Retrieve the capitalized name from the loaded data."""
        return self._config_data.get("name", self.registry_name)

    def show(self) -> list:
        """Show all the data."""
        return [self.name, self.version, self.description]

    def apply_to_payload(self, messages: list[dict]) -> dict:
        """Construct a complete API payload containing the system prompt and options."""
        full_messages = [
            {"role": "system", "content": self.system_prompt},
            *messages,
        ]
        return {
            "messages": full_messages,
        }
