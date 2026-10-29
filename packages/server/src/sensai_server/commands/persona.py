from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, override

from sensai_protocol.error import SensaiError

from sensai_server.commands.base import Command
from sensai_server.llm.persona import read_persona_file

if TYPE_CHECKING:
    from collections.abc import Callable, Coroutine
    from typing import Any

    from sensai_server.llm.persona import Persona
    from sensai_server.session import Session, SessionRegistry

# The fields `/persona update` can set, and the keyword each one maps onto in the manager.
UPDATABLE_FIELDS = {"name": "new_name", "description": "description", "prompt": "prompt"}
# The sub-commands whose argument is the name of a stored persona.
NAMED_SUBCOMMANDS = ("info", "load", "update")


class PersonaArgumentError(SensaiError):
    """Raised when an argument of the persona command cannot be read."""

    def __init__(self, reason: str) -> None:
        """Build the error from the `reason` the argument was rejected."""
        self.reason = reason
        super().__init__(f"[PersonaArgumentError] {reason}")


class PersonaCommand(Command):
    """Persona command class to list, describe, load, create and update personas.

    The persona manager is the session's. Its database calls run in a thread, so that they do not block the other
    clients, and its writes are shielded, so that a stopped command does not leave a write half done.
    """

    name = "persona"
    help = "List, describe, load, unload, create or update the personas."
    usage = (
        "/persona list"
        " | info <name>"
        " | load <name>"
        " | unload"
        " | create <path>"
        " | update <name> <field>=<value> ..."
        " | updateFromFile <path>"
    )
    _sessions: SessionRegistry
    _subcommands: dict[str, tuple[Callable[..., Coroutine[Any, Any, None]], int, int | None]]

    def __init__(self, sessions: SessionRegistry) -> None:
        """Initialize the Persona command with the `sessions` to refresh when a persona is updated."""
        super().__init__()
        self._sessions = sessions
        self._subcommands = {
            "list": (self._list, 0, 0),
            "info": (self._info, 1, 1),
            "load": (self._load, 1, 1),
            "unload": (self._unload, 0, 0),
            "create": (self._create, 1, 1),
            "update": (self._update, 2, None),
            "updateFromFile": (self._update_from_file, 1, 1),
        }

    @override
    async def execute(self, session: Session, *args: str) -> None:
        """Run the sub-command in `args`, reporting any failure to the client instead of raising."""
        verb, *rest = args or ("",)
        subcommand = self._subcommands.get(verb)
        if subcommand is None:
            await session.out.send_system_message(f"Usage: {self.usage}")
            return
        handler, min_args, max_args = subcommand
        if len(rest) < min_args or (max_args is not None and len(rest) > max_args):
            await session.out.send_system_message(f"Usage: {self.usage}")
            return
        try:
            await handler(session, *rest)
        except SensaiError as err:
            await session.out.send_system_message(err.message)

    @override
    def completions(self, session: Session, args: list[str]) -> list[str]:
        """Return the sub-commands, then the persona names after a sub-command taking one, then the fields."""
        if not args:
            return sorted(self._subcommands)
        if len(args) == 1 and args[0] in NAMED_SUBCOMMANDS:
            return sorted(persona.name for persona in session.persona_manager.get_personas())
        if len(args) >= 2 and args[0] == "update":  # noqa: PLR2004 - `update <name>` then the fields
            return [f"{field}=" for field in sorted(UPDATABLE_FIELDS)]
        return []

    async def _list(self, session: Session) -> None:
        """Display every stored persona, with its description, marking the loaded one."""
        personas = await asyncio.to_thread(session.persona_manager.get_personas)
        if not personas:
            await session.out.send_system_message("No persona is stored. Create one with: /persona create <path>")
            return
        lines = ["Stored personas:"]
        lines.extend(
            f"- {persona.name}{' (loaded)' if self._is_loaded(session, persona) else ''}: {persona.description}"
            for persona in sorted(personas, key=lambda persona: persona.name)
        )
        await session.out.send_system_message("\n".join(lines))

    async def _info(self, session: Session, name: str) -> None:
        """Display everything known about the persona named `name`, its prompt included."""
        persona = session.persona_manager.get_persona(name)
        lines = [
            f"Name: {persona.name}",
            f"Description: {persona.description}",
            f"Loaded: {'yes' if self._is_loaded(session, persona) else 'no'}",
        ]
        if persona.created_at is not None:
            lines.append(f"Created: {persona.created_at:%Y-%m-%d %H:%M}")
        lines.extend(("Prompt:", persona.prompt))
        await session.out.send_system_message("\n".join(lines))

    async def _load(self, session: Session, name: str) -> None:
        """Apply the persona named `name` to the session's conversation, and save it on its branch, if any."""
        manager = session.persona_manager
        persona = await asyncio.to_thread(manager.get_persona, name)
        session.conv.persona = persona
        branch = session.conv.uuid
        if branch is not None and not await asyncio.shield(
            asyncio.to_thread(manager.set_branch_persona, branch, persona)
        ):
            await session.out.send_system_message(f"Persona '{persona.name}' loaded, but not saved on the branch.")
            return
        await session.out.send_system_message(f"Persona '{persona.name}' loaded.", append_response=True)

    async def _unload(self, session: Session) -> None:
        """Remove the persona applied to the session's conversation. The branch keeps the one it saved."""
        persona = session.conv.persona
        if persona is None:
            await session.out.send_system_message("No persona loaded.")
            return
        session.conv.persona = None
        await session.out.send_system_message(f"Persona '{persona.name}' unloaded.")

    async def _create(self, session: Session, path: str) -> None:
        """Store the persona described by the file at `path`, read on the client."""
        draft = await read_persona_file(session.out, path)
        persona = await asyncio.shield(asyncio.to_thread(session.persona_manager.create_persona, draft))
        await session.out.send_system_message(f"Persona '{persona.name}' created from '{path}'.")

    async def _update(self, session: Session, name: str, *assignments: str) -> None:
        """Set the `field=value` pairs in `assignments` on the persona named `name`."""
        fields = self._parse_assignments(assignments)
        changes = {UPDATABLE_FIELDS[field]: value for field, value in fields.items()}
        persona = await asyncio.shield(
            asyncio.to_thread(lambda: session.persona_manager.update_persona(name, **changes))
        )
        self._sessions.refresh_persona(persona)
        await session.out.send_system_message(f"Persona '{persona.name}' updated: {', '.join(fields)}.")

    async def _update_from_file(self, session: Session, path: str) -> None:
        """Update the persona the file at `path`, read on the client, names, with its description and prompt."""
        draft = await read_persona_file(session.out, path)
        persona = await asyncio.shield(asyncio.to_thread(session.persona_manager.update_persona_from_draft, draft))
        self._sessions.refresh_persona(persona)
        await session.out.send_system_message(f"Persona '{persona.name}' updated from '{path}'.")

    @staticmethod
    def _is_loaded(session: Session, persona: Persona) -> bool:
        """Return whether `persona` is the one applied to the session's conversation."""
        loaded = session.conv.persona
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
