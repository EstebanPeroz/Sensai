from typing import TYPE_CHECKING, override

from sensai.commands.base import Command
from sensai.error import SensaiError
from sensai.llm.persona import Persona

if TYPE_CHECKING:
    from collections.abc import Callable

    from sensai.llm.persona_manager import PersonaManager
    from sensai.ui.adapter import UIAdapter

# The fields `/persona update` can set, and the keyword each one maps onto in the manager.
UPDATABLE_FIELDS = {"name": "new_name", "description": "description", "prompt": "prompt"}


class PersonaArgumentError(SensaiError):
    """Raised when an argument of the persona command cannot be read."""

    def __init__(self, reason: str) -> None:
        """Build the error from the `reason` the argument was rejected."""
        self.reason = reason
        super().__init__(f"[PersonaArgumentError] {reason}")


class PersonaCommand(Command):
    """Persona command class to list, describe, load, create and update personas."""

    name = "persona"
    help = "List, describe, load, create or update the personas."
    usage = (
        "/persona list"
        " | info <name>"
        " | load <name>"
        " | create <path>"
        " | update <name> <field>=<value> ..."
        " | updateFromFile <path>"
    )
    _persona_manager: PersonaManager
    _subcommands: dict[str, tuple[Callable[..., None], int, int | None]]

    def __init__(self, ui: UIAdapter, persona_manager: PersonaManager) -> None:
        """Initialize the Persona command with the manager holding the personas."""
        super().__init__(ui)
        self._persona_manager = persona_manager
        self._subcommands = {
            "list": (self._list, 0, 0),
            "info": (self._info, 1, 1),
            "load": (self._load, 1, 1),
            "create": (self._create, 1, 1),
            "update": (self._update, 2, None),
            "updateFromFile": (self._update_from_file, 1, 1),
        }

    @override
    def execute(self, *args: str) -> None:
        """Run the sub-command in `args`, reporting any failure to the UI instead of raising."""
        verb, *rest = args or ("",)
        subcommand = self._subcommands.get(verb)
        if subcommand is None:
            self._ui.send_system_message(f"Usage: {self.usage}")
            return
        handler, min_args, max_args = subcommand
        if len(rest) < min_args or (max_args is not None and len(rest) > max_args):
            self._ui.send_system_message(f"Usage: {self.usage}")
            return
        try:
            handler(*rest)
        except SensaiError as err:
            self._ui.send_system_message(err.message)

    @override
    def completions(self) -> list[str]:
        """Return the sub-commands, which is the argument the UI can complete."""
        return sorted(self._subcommands)

    def _list(self) -> None:
        """Display every stored persona, with its description, marking the loaded one."""
        personas = self._persona_manager.get_personas()
        if not personas:
            self._ui.send_system_message("No persona is stored. Create one with: /persona create <path>")
            return
        lines = ["Stored personas:"]
        lines.extend(
            f"- {persona.name}{' (loaded)' if self._is_loaded(persona) else ''}: {persona.description}"
            for persona in sorted(personas, key=lambda persona: persona.name)
        )
        self._ui.send_system_message("\n".join(lines))

    def _info(self, name: str) -> None:
        """Display everything known about the persona named `name`, its prompt included."""
        persona: Persona = self._persona_manager.get_persona(name)
        lines = [
            f"Name: {persona.name}",
            f"Description: {persona.description}",
            f"Loaded: {'yes' if self._is_loaded(persona) else 'no'}",
        ]
        if persona.created_at is not None:
            lines.append(f"Created: {persona.created_at:%Y-%m-%d %H:%M}")
        lines.extend(("Prompt:", persona.prompt))
        self._ui.send_system_message("\n".join(lines))

    def _load(self, name: str) -> None:
        """Load the persona named `name` from the database and apply it to the session."""
        persona = self._persona_manager.activate(name)
        self._ui.send_system_message(f"Persona '{persona.name}' loaded.", append_response=True)

    def _create(self, path: str) -> None:
        """Store the persona described by the file at `path`."""
        persona = self._persona_manager.create_persona_from_file(path)
        self._ui.send_system_message(f"Persona '{persona.name}' created from '{path}'.")

    def _update(self, name: str, *assignments: str) -> None:
        """Set the `field=value` pairs in `assignments` on the persona named `name`."""
        fields = self._parse_assignments(assignments)
        changes = {UPDATABLE_FIELDS[field]: value for field, value in fields.items()}
        persona = self._persona_manager.update_persona(name, **changes)
        self._ui.send_system_message(f"Persona '{persona.name}' updated: {', '.join(fields)}.")

    def _update_from_file(self, path: str) -> None:
        """Update the persona the file at `path` names, with the description and prompt it holds."""
        draft = Persona.from_file(path)
        persona = self._persona_manager.update_persona(
            draft.name,
            description=draft.description,
            prompt=draft.prompt,
        )
        self._ui.send_system_message(f"Persona '{persona.name}' updated from '{path}'.")

    def _is_loaded(self, persona: Persona) -> bool:
        """Return whether `persona` is the one applied to the session."""
        loaded = self._persona_manager.active_persona()
        return loaded is not None and loaded.uuid == persona.uuid

    @staticmethod
    def _parse_assignments(assignments: tuple[str, ...]) -> dict[str, str]:
        """Read the `field=value` pairs of `assignments`, keyed by field."""
        fields: dict[str, str] = {}
        current: str | None = None
        for token in assignments:
            field, separator, value = token.partition("=")
            if not separator:
                if current is None:
                    reason = f"expected 'field=value', got '{token}'"
                    raise PersonaArgumentError(reason)
                fields[current] += f" {token}"
                continue
            if field not in UPDATABLE_FIELDS:
                reason = f"cannot set '{field}', expected one of: {', '.join(sorted(UPDATABLE_FIELDS))}"
                raise PersonaArgumentError(reason)
            if field in fields:
                reason = f"'{field}' is set twice"
                raise PersonaArgumentError(reason)
            fields[field] = value
            current = field
        return {field: value.strip() for field, value in fields.items()}
