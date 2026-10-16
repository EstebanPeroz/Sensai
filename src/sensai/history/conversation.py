import threading
from typing import TYPE_CHECKING

from sensai.error import SensaiError
from sensai.llm.message import Message, Role
from sensai.mcp.request import add_tool_result, add_tool_schemas, execute_one_tool_call
from sensai.memory.history import History
from sensai.memory.persistent_db import get_db

if TYPE_CHECKING:
    from collections.abc import Iterator
    from uuid import UUID

    from sensai.llm.persona_manager import PersonaManager
    from sensai.llm.provider_manager import ProviderManager
    from sensai.llm.responses import ChatResponse
    from sensai.tools.tool_manager import ToolManager
    from sensai.ui.adapter import UIAdapter


class Conversation:
    """Drives a chat exchange with a provider, streaming its response back to a UI."""

    _provider_manager: ProviderManager
    _persona_manager: PersonaManager
    history: History
    _tool_manager: ToolManager
    _pending_tool_result: dict | None = None
    uuid: UUID | None = None

    def __init__(
        self,
        provider_manager: ProviderManager,
        persona_manager: PersonaManager,
        tool_manager: ToolManager,
    ) -> None:
        """Init method to create the conversation callss."""
        self._provider_manager = provider_manager
        self.history = History()
        self._persona_manager = persona_manager
        self._tool_manager = tool_manager

    def add_conv_to_db(self, uuid: UUID | None = None, name: str = "Branch") -> None:
        """Init the conversation in the DB if first message."""
        db = get_db()
        if db is None:
            return
        self.uuid = db.create_branch(uuid, name)
        if self.uuid:
            db.set_branch_model(self.uuid, self._provider_manager.provider().current_model())

    def add_message(self, role: Role, content: str, ui: UIAdapter) -> bool:
        """Add message to History and db."""
        db = get_db()
        if not self.uuid or not db:
            ui.send_system_message("Failed to add message")
            return False

        try:
            uuid = db.add_message_to_branch(self.uuid, role, content)
        except SensaiError as err:
            ui.send_system_message(err.message)
            return False

        if not uuid:
            ui.send_system_message("Failed to add message")
            return False

        self.history.insert(Message(uuid=uuid, role=role, content=content))
        return True

    def _stream_follow_up(
        self,
        payload: dict,
        tool_call: dict,
        tool_result: dict | None,
        stop_streaming: threading.Event,
        ui: UIAdapter,
    ) -> str:
        """Stream the one allowed follow-up request after a tool call."""
        follow_up = add_tool_result(payload, tool_call, tool_result)
        chunks = self._provider_manager.provider().chat(follow_up, stream=True)
        content = ""
        if chunks is None:
            return content
        for chunk in chunks:
            if stop_streaming.is_set():
                return content
            if chunk.content is not None:
                content += chunk.content
            ui.send_ai_response(chunk)
        return content

    def _stream_chunks(
        self,
        chunks: Iterator[ChatResponse],
        stop_streaming: threading.Event,
        stream_done: threading.Event,
        ui: UIAdapter,
        payload: dict,
    ) -> None:
        """Forward chunks to the UI until they run out or a stop is requested."""
        try:
            content: str = ""
            for chunk in chunks:
                if stop_streaming.is_set():
                    return
                if chunk.content is not None:
                    content += chunk.content
                ui.send_ai_response(chunk)

                if chunk.tool_calls:
                    tool_call = chunk.tool_calls[0]
                    self._pending_tool_result = execute_one_tool_call(chunk.tool_calls, self._tool_manager)
                    content += self._stream_follow_up(
                        payload,
                        tool_call,
                        self._pending_tool_result,
                        stop_streaming,
                        ui,
                    )
                    break
                if chunk.error is not None:
                    break
            self.add_message(Role.ASSISTANT, content, ui)
        except SensaiError as err:
            ui.send_error(str(err))
        finally:
            stream_done.set()

    def chat(self, user_input: str, ui: UIAdapter) -> None:
        """Send `user_input` to the provider and stream its response to `ui` until done or interrupted."""
        ui.send_input(Role.USER, user_input)
        if self.add_message(Role.USER, user_input, ui) is False:
            return
        messages = self.history.to_json()
        persona = self._persona_manager.active_persona()
        if persona is not None:
            messages = [persona.to_payload(), *messages]
        payload: dict = {"model": self._provider_manager.provider().current_model(), "messages": messages}
        add_tool_schemas(payload, self._tool_manager)
        chunks = self._provider_manager.provider().chat(payload, stream=True)
        if chunks is None:
            return

        stop_streaming = threading.Event()
        stream_done = threading.Event()
        threading.Thread(
            target=self._stream_chunks,
            args=(chunks, stop_streaming, stream_done, ui, payload),
            daemon=True,
        ).start()

        while not stream_done.is_set():
            if not ui.wait_event(timeout=0.05):
                continue
            stop_streaming.set()
            break
