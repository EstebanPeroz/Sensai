from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sensai_server.session import Session


async def chat(session: Session, prompt: str) -> None:
    """Chat loop on the Conversation."""
    await session.out.send_system_message(prompt)
