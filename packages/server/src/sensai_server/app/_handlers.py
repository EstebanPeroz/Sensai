from __future__ import annotations

from typing import TYPE_CHECKING

from sensai_protocol import Role

from sensai_server.chat import chat

if TYPE_CHECKING:
    from collections.abc import Sequence

    from sensai_protocol import (
        Cancel,
        Chat,
        Command,
        CommandContext,
        Complete,
        CompressionAnswer,
        ConfirmAnswer,
        Login,
        Logout,
        UserAnswer,
    )

    from sensai_server.app.outbound import Outbound
    from sensai_server.commands import CommandRegistry
    from sensai_server.events import EventBus
    from sensai_server.memory.persistent_db import PersistentDatabase
    from sensai_server.pipeline import ChatPipeline
    from sensai_server.rag import Rag
    from sensai_server.session import Session, SessionRegistry

PLACEHOLDER_DELAY = 0.3


async def handle_login(
    registry: SessionRegistry, out: Outbound, session: Session | None, message: Login, commands: Sequence[str]
) -> Session | None:
    """Check that a provider serves `message.model`, then open a session and answer with `session`.

    `commands` are the typed names of the server's commands, sent for the client's completion.

    An unknown model is answered with an `error` listing the available models, and the socket stays open. A login
    on a connection that already has a session is refused, and the session is kept. Returns the connection's
    session.
    """
    if session is not None:
        await out.send_error(f"Already logged in with {session.conv.model}.")
        return session
    return await registry.open(out, message.model, commands)


async def handle_logout(registry: SessionRegistry, session: Session, _: Logout) -> None:
    """Stop the session's running task and free it from `registry`. The endpoint then closes the socket."""
    await registry.close(session.client_id)


async def _stop_client_command(session: Session) -> None:
    """Stop the session's task if it waits for a client command: a new chat or command replaces it."""
    if session.busy and session.out.waiting_client_command:
        label = session.task_label
        await session.stop()
        stopped = f"Command '{label}'" if label is not None else "Task"
        await session.out.send_system_message(f"{stopped} stopped: a new message was sent.")


async def handle_chat(  # noqa: PLR0913, PLR0917 - the shared services a chat needs come through here
    session: Session,
    message: Chat,
    pipeline: ChatPipeline,
    db: PersistentDatabase,
    events: EventBus,
    rag: Rag | None = None,
) -> None:
    """Echo the prompt as `input`, then start the answer as the session's task, streamed as `ai_chunk`.

    The prompt goes through `pipeline`, and the prompt and its answer are saved in `db`. The branch the first prompt
    creates gets its collection in `rag`, if any. Every call to the model is emitted on `events`. A task waiting for
    a client command is stopped first. Refused with an `error` while another task is running.
    """
    await _stop_client_command(session)
    if session.busy:
        await session.out.send_error("An answer is already running: cancel it first.")
        return
    await session.out.send_input(Role.USER, message.prompt)
    session.start(chat(session, message.prompt, pipeline, db, events, rag))


async def handle_command(session: Session, message: Command, command_registry: CommandRegistry) -> None:
    """Echo the command as `input`, then run it against the session's commands as the session's task.

    A task waiting for a client command is stopped first. Refused with an `error` while another task is running.
    """
    await _stop_client_command(session)
    if session.busy:
        await session.out.send_error("A task is already running: cancel it first.")
        return
    await session.out.send_input(Role.USER, message.command, command=True)
    session.start(command_registry.execute(session, message.command), message.command)


async def handle_cancel(session: Session, _: Cancel) -> None:
    """Cancel the session's running task, if any. Nothing is sent back.

    It waits until the task has stopped, so a `chat` sent right after the `cancel` is not refused as busy.
    """
    await session.stop()


async def handle_complete(session: Session, message: Complete, commands: CommandRegistry) -> None:
    """Answer with the `completions` of the word being typed in `message.text`, after its finished words."""
    await session.out.send_completions(message.text, commands.completions(session, message.text))


async def handle_confirm_answer(session: Session, message: ConfirmAnswer) -> None:
    """Give the answer to the waiting `confirm_tool`. It is not refused while busy: the busy task is the one waiting."""
    session.out.resolve(message.request_id, message.answer)


async def handle_user_answer(session: Session, message: UserAnswer) -> None:
    """Give the answer to the waiting `ask_user`. It is not refused while busy: the busy task is the one waiting."""
    session.out.resolve(message.request_id, message.answer)


async def handle_compression_answer(session: Session, message: CompressionAnswer) -> None:
    """Give the answer to the waiting `ask_compression`.

    It is not refused while busy: the busy task is the one waiting.
    """
    session.out.resolve(message.request_id, message.answer)


async def handle_command_context(session: Session, message: CommandContext) -> None:
    """Give the answer to the waiting `request_command`. It is not refused while busy: the busy task is the one waiting.

    An answer to no pending question, such as the late answer to a stopped command, is dropped.
    """
    session.out.resolve(message.request_id, message)
