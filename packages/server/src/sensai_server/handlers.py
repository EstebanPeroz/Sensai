from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sensai_protocol import Cancel, Chat, Command, Complete, Login, Logout

    from sensai_server.outbound import Outbound


async def handle_login(out: Outbound, message: Login) -> None:
    """Check that a provider serves `message.model`, then open a session and answer with `session`.

    An unknown model is answered with an `error` listing the available models, and the socket stays open.
    """


async def handle_logout(out: Outbound, message: Logout) -> None:
    """Cancel the session's running task and remove it from the registry. The endpoint then closes the socket."""


async def handle_chat(out: Outbound, message: Chat) -> None:
    """Echo the prompt as `input`, then start the answer as the session's task, streamed as `ai_chunk`.

    Refused with an `error` while a task is already running.
    """


async def handle_command(out: Outbound, message: Command) -> None:
    """Echo the command as `input`, then run it against the session's commands as the session's task.

    An unknown command is answered with a `system` message.
    """


async def handle_cancel(out: Outbound, message: Cancel) -> None:
    """Cancel the session's running task, if any. Nothing is sent back."""


async def handle_complete(out: Outbound, message: Complete) -> None:
    """Answer with the `completions` of the finished words of `message.text`."""
