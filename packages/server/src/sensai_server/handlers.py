from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from sensai_protocol import AiChunk, Role

if TYPE_CHECKING:
    from collections.abc import Callable, Coroutine

    from sensai_protocol import Cancel, Chat, Command, Complete, ConfirmAnswer, Login, Logout, UserAnswer

    from sensai_server.outbound import Outbound
    from sensai_server.session import Session, SessionRegistry

type ChatRunner = Callable[[Session, str], Coroutine[Any, Any, None]]
type CommandRunner = Callable[[Session, str], Coroutine[Any, Any, None]]

PLACEHOLDER_DELAY = 0.3


# TEMPORARY: replaced by the conversation's chat once it is copied from `src/`.
async def placeholder_chat(session: Session, prompt: str) -> None:
    """Stream the words of `prompt` back as the answer, until the conversation is copied from `src/`."""
    for word in prompt.split():
        await session.out.send_ai_response(AiChunk(content=f"{word} "))
        await asyncio.sleep(PLACEHOLDER_DELAY)
    await session.out.send_ai_response(AiChunk(done=True))


# TEMPORARY: replaced by the session's command registry once the commands are copied from `src/`.
async def placeholder_command(session: Session, command: str) -> None:
    """Report that commands do not exist yet, until they are copied from `src/`."""
    await session.out.send_system_message(f"Commands are not available yet: {command}")


async def handle_login(
    registry: SessionRegistry, out: Outbound, session: Session | None, message: Login
) -> Session | None:
    """Check that a provider serves `message.model`, then open a session and answer with `session`.

    An unknown model is answered with an `error` listing the available models, and the socket stays open. A login
    on a connection that already has a session is refused, and the session is kept. Returns the connection's
    session.
    """
    if session is not None:
        await out.send_error(f"Already logged in with {session.model}.")
        return session
    return await registry.open(out, message.model)


async def handle_logout(registry: SessionRegistry, session: Session, _: Logout) -> None:
    """Stop the session's running task and free it from `registry`. The endpoint then closes the socket."""
    await registry.close(session.client_id)


async def handle_chat(session: Session, message: Chat, run: ChatRunner) -> None:
    """Echo the prompt as `input`, then start the answer as the session's task, streamed as `ai_chunk`.

    Refused with an `error` while a task is already running.
    """
    if session.busy:
        await session.out.send_error("An answer is already running: cancel it first.")
        return
    await session.out.send_input(Role.USER, message.prompt)
    session.start(run(session, message.prompt))


async def handle_command(session: Session, message: Command, run: CommandRunner) -> None:
    """Echo the command as `input`, then run it against the session's commands as the session's task.

    Refused with an `error` while a task is already running.
    """
    if session.busy:
        await session.out.send_error("A task is already running: cancel it first.")
        return
    await session.out.send_input(Role.USER, message.command, command=True)
    session.start(run(session, message.command))


async def handle_cancel(session: Session, _: Cancel) -> None:
    """Cancel the session's running task, if any. Nothing is sent back.

    It waits until the task has stopped, so a `chat` sent right after the `cancel` is not refused as busy.
    """
    await session.stop()


# TEMPORARY: always answers no values, until the commands' completion sources are copied from `src/`.
async def handle_complete(session: Session, message: Complete) -> None:
    """Answer with the `completions` of the finished words of `message.text`: none until commands are copied."""
    await session.out.send_completions(message.text, [])


async def handle_confirm_answer(session: Session, message: ConfirmAnswer) -> None:
    """Give the answer to the waiting `confirm_tool`. It is not refused while busy: the busy task is the one waiting."""
    session.out.resolve(message.request_id, message.answer)


async def handle_user_answer(session: Session, message: UserAnswer) -> None:
    """Give the answer to the waiting `ask_user`. It is not refused while busy: the busy task is the one waiting."""
    session.out.resolve(message.request_id, message.answer)
