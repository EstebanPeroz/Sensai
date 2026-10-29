from __future__ import annotations

from typing import TYPE_CHECKING, cast

from sensai_protocol import ClientCommand, FileContent, ReadFileParams

from sensai_server.llm.persona.error import PersonaFileError
from sensai_server.llm.persona.persona import Persona
from sensai_server.outbound import ClientCommandError

if TYPE_CHECKING:
    from sensai_server.outbound import Outbound


async def read_persona_file(out: Outbound, path: str) -> Persona:
    """Ask the client for the file at `path` and parse its content into an unsaved persona.

    The client only reads the file: the server parses it. Raises PersonaFileError when the file cannot be read or
    does not describe a persona.
    """
    try:
        file = await out.request_command(ClientCommand.READ_FILE, ReadFileParams(path=path))
    except ClientCommandError as err:
        raise PersonaFileError(path, err.reason) from err
    return Persona.from_toml(cast("FileContent", file).content, path)
