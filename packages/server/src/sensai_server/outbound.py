from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any
from uuid import UUID, uuid4

from fastapi import WebSocket, WebSocketDisconnect
from sensai_protocol import (
    AskUser,
    ClientCommand,
    Completions,
    Confirmation,
    ConfirmTool,
    Dashboard,
    Effort,
    Error,
    History,
    Input,
    ProtocolError,
    RequestCommand,
    System,
    parse_ctx,
)
from sensai_protocol.error import SensaiError

if TYPE_CHECKING:
    from sensai_protocol import (
        AiChunk,
        CommandContext,
        DashboardReport,
        HistoryMessage,
        ProtocolModel,
        Role,
        ServerMessage,
    )


class ClientCommandError(SensaiError):
    """Raised when the client did not run a client command: no answer, a wrong one, or the error it failed with."""

    def __init__(self, command: ClientCommand, reason: str) -> None:
        """Build the error from the `command` asked and the `reason` it was not run."""
        self.command = command
        self.reason = reason
        super().__init__(f"[ClientCommandError] {command}: {reason}")


class Outbound:
    """Sends the server messages of one connection to its socket.

    Replaces `UIAdapter` on the server side and keeps its method names, so the code copied from `src/` keeps its
    shape.
    """

    _ws: WebSocket
    _pending: dict[UUID, asyncio.Future[Any]]
    _client_requests: set[UUID]

    def __init__(self, ws: WebSocket) -> None:
        """Send on `ws`, an accepted websocket."""
        self._ws = ws
        self._pending = {}
        self._client_requests = set()

    @property
    def waiting(self) -> bool:
        """Whether a question waits for the client's answer."""
        return bool(self._pending)

    async def send(self, message: ServerMessage) -> bool:
        """Send `message`. On a closed socket it is dropped: the receive loop sees the disconnection.

        Returns whether it was sent.
        """
        try:
            await self._ws.send_text(message.model_dump_json())
        except WebSocketDisconnect, RuntimeError:
            return False
        return True

    async def send_ai_response(self, chunk: AiChunk) -> None:
        """Send a chunk of the streamed answer."""
        await self.send(chunk)

    async def send_error(self, message: str) -> None:
        """Send an error to display."""
        await self.send(Error(message=message))

    async def send_input(self, role: Role, content: str, *, command: bool = False) -> None:
        """Echo an input of the conversation, or a typed `command`."""
        await self.send(Input(role=role, content=content, command=command))

    async def send_system_message(self, message: str, *, append_response: bool = False) -> None:
        """Send an application message, such as a command output, appended to the previous one if asked."""
        await self.send(System(message=message, append=append_response))

    async def load_conversation(self, messages: list[HistoryMessage]) -> None:
        """Send a conversation that replaces the client's transcript."""
        await self.send(History(messages=messages))

    async def update_thinking_effort(self, level: str) -> None:
        """Send the current thinking effort `level` of the model."""
        await self.send(Effort(level=level))

    async def send_completions(self, text: str, values: list[str]) -> None:
        """Answer a `complete` of `text` with its completion `values`."""
        await self.send(Completions(text=text, values=values))

    async def show_dashboard(self, report: DashboardReport) -> None:
        """Send a monitoring report, shown until the user closes it."""
        await self.send(Dashboard(report=report))

    async def ask_confirmation(self, tool_name: str, arguments: dict[str, Any]) -> Confirmation:
        """Ask whether the tool `tool_name` may run with `arguments`, waiting for the client's `confirm_answer`.

        Denied when the question cannot be sent.
        """
        request_id = uuid4()
        message = ConfirmTool(request_id=request_id, tool_name=tool_name, arguments=arguments)
        answer = await self._ask(request_id, message)
        return Confirmation.DENY if answer is None else answer

    async def ask_user(self, question: str) -> str | None:
        """Ask `question` on behalf of the model, waiting for the client's `user_answer`.

        The question and its answer are then shown in the chat. Returns None when the user skips it or the question
        cannot be sent.
        """
        request_id = uuid4()
        answer = await self._ask(request_id, AskUser(request_id=request_id, question=question))
        await self.send_system_message(f"Sensai asked: {question}\n> {answer or '(no answer)'}")
        return answer

    async def request_command(self, command: ClientCommand, params: ProtocolModel) -> ProtocolModel:
        """Ask the client to run `command` with `params`, waiting for its `command_context`.

        Returns the context of the command, checked against its model. Raises ClientCommandError when the client
        does not answer, answers for another command, reports an error, or sends a context that does not match.
        """
        request_id = uuid4()
        message = RequestCommand(request_id=request_id, command=command, params=params.model_dump())
        self._client_requests.add(request_id)
        try:
            answer: CommandContext | None = await self._ask(request_id, message)
        finally:
            self._client_requests.discard(request_id)
        if answer is None:
            raise ClientCommandError(command, "the client did not answer")
        if answer.command != command:
            raise ClientCommandError(command, f"the client answered for '{answer.command}'")
        if answer.error is not None:
            raise ClientCommandError(command, answer.error)
        try:
            return parse_ctx(command, answer.ctx)
        except ProtocolError as err:
            raise ClientCommandError(command, f"invalid answer ({err})") from err

    def resolve(self, request_id: UUID, answer: object) -> None:
        """Give `answer` to the question of `request_id`. An answer to no pending question is dropped."""
        future = self._pending.get(request_id)
        if future is not None and not future.done():
            future.set_result(answer)

    async def _ask(self, request_id: UUID, message: ServerMessage) -> Any:  # noqa: ANN401 - the answer's type depends on the question
        """Send `message` and wait for the answer to `request_id`, None if it cannot be sent.

        The question is forgotten once answered, or when the waiting task is cancelled.
        """
        future: asyncio.Future[Any] = asyncio.get_running_loop().create_future()
        self._pending[request_id] = future
        try:
            if not await self.send(message):
                return None
            return await future
        finally:
            del self._pending[request_id]
