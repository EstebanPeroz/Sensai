from typing import TYPE_CHECKING

from sensai_protocol.error import SensaiError
from sensai_protocol.server_messages import Role

if TYPE_CHECKING:
    from sensai_server.session import Session


async def chat(session: Session, prompt: str) -> None:
    """Chat loop on the Conversation."""
    try:
        await _run_chat(session, prompt)
    except SensaiError as err:
        await session.out.send_error(str(err))


async def _run_chat(session: Session, prompt: str) -> None:
    session.out.send_input(Role.USER, prompt)
