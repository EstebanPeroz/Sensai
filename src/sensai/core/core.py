import contextlib
from typing import TYPE_CHECKING

import anyio

from sensai import ui
from sensai.commands import Ask, Branch, Command, Help, Index, Model, PersonaCommand, StateCommand
from sensai.commands.data_transfer import Export, Import
from sensai.commands.effort import Effort
from sensai.commands.registry import CommandRegistry, UnknownCommandError
from sensai.commands.tools import ToolsCommand
from sensai.error import SensaiError
from sensai.history.conversation import Conversation
from sensai.llm.embedder import Embedder
from sensai.llm.error import ModelError
from sensai.llm.message import Role
from sensai.llm.persona import DuplicatePersonaError
from sensai.llm.persona_manager import PersonaManager
from sensai.llm.provider_manager import ProviderManager
from sensai.llm.thinking.react import ReAct
from sensai.mcp.client import MCPClient
from sensai.memory.persistent_db.base import get_db
from sensai.memory.persistent_db.error import DbConnectionError
from sensai.rag.collection import Collection
from sensai.rag.indexation_rag import DEFAULT_EMBEDDING_MODEL, Indexation
from sensai.rag.indexer import Indexer
from sensai.rag.retrievial_rag import Retrieval
from sensai.tools import build_tools
from sensai.tools.tool_manager import ToolManager
from sensai.tools.web import web_tools

if TYPE_CHECKING:
    import argparse

    from sensai.config.settings import Settings
    from sensai.memory.persistent_db.adapter import PersistentDatabase
    from sensai.tools.tool import Tool


class Core:
    """Wires a UI to a conversation, dispatching UI events until the user quits."""

    conversation: Conversation
    _provider_manager: ProviderManager
    _react: ReAct
    _ui: ui.UIAdapter
    _command_registry: CommandRegistry
    _settings: Settings
    _mcp_client: MCPClient | None
    _tool_manager: ToolManager

    def __init__(self, ui: ui.UIAdapter, settings: Settings, args: argparse.Namespace) -> None:
        """Set up the provider manager and conversation. Safe to call before `ui.run()` starts."""
        self._ui = ui
        self._settings = settings
        self._mcp_client = MCPClient(settings.mcp) if settings.mcp is not None else None

        db = get_db()
        if db is None:
            msg = "no persistent database"
            raise DbConnectionError(msg)
        self._persona_manager = PersonaManager(db)
        with contextlib.suppress(DuplicatePersonaError):
            self._persona_manager.create_default_persona()
        self._provider_manager = ProviderManager(settings.llm)
        if not (self._provider_manager.set_model(args.model)):
            lines = [f"Model '{args.model}' not found in provider manager.", "Available models:"]
            lines.extend(f"- {model}" for model in sorted(self._provider_manager.get_models()))
            msg = "\n".join(lines)
            raise ModelError(msg)
        tools = [*build_tools(), *self._build_web_tools()]
        if self._mcp_client is not None:
            tools.extend(anyio.run(self._mcp_client.start))
        self._tool_manager = ToolManager(tools, disabled=settings.features.disabled_in("tools"))
        self._react = ReAct(self._provider_manager, self._tool_manager)
        self.conversation = Conversation(self._provider_manager, self._persona_manager, self._react)
        self._command_registry = self._build_command_registry(db)

    def run(self) -> None:
        """Run the application and close the MCP session when it exits."""
        try:
            self._run_application()
        finally:
            if self._mcp_client is not None:
                anyio.run(self._mcp_client.close)

    def _run_application(self) -> None:
        """Wait for `ui` to be ready, then dispatch its events to the conversation until quit.

        Meant to run off the main thread, since it blocks on `ui` being open, which
        requires `ui.run()` to be driving the UI's event loop on the main thread.
        """
        self._ui.open()

        unknown = self._tool_manager.unknown_disabled()
        if unknown:
            self._ui.send_system_message(f"Unknown disabled tools in the settings: {', '.join(unknown)}")

        self._ui.set_completions(
            {f"/{command.name}": command.completions for command in self._command_registry.commands()}
        )
        self._ui.update_thinking_effort(Effort.effort_level(self._react.max_iterations))

        event: ui.Event | None = None
        while not self._is_quit(event):
            self._ui.wait_event()
            event = self._ui.get_event()
            if event is None:
                continue
            if self._is_quit(event):
                break
            if event.type == ui.EventType.UserContent:
                self._launch_conv_chat(event.content)
            elif event.type == ui.EventType.Command:
                self._launch_command(event.content)

        self._ui.close()

    def _build_command_registry(self, db: PersistentDatabase) -> CommandRegistry:
        """Build a command registry with all available commands."""
        Command.bind_ui(self._ui)
        StateCommand.bind_db(db)
        indexer, retrieval = self._build_rag()
        registry = CommandRegistry(
            [
                Model(self._provider_manager),
                PersonaCommand(self._persona_manager),
                Branch(self.conversation, self._provider_manager),
                ToolsCommand(self._tool_manager),
                Effort(self._react),
                Export(self.conversation),
                Import(),
                Index(indexer),
                Ask(retrieval),
            ]
        )
        registry.register_command(Help(registry))
        return registry

    def _build_rag(self) -> tuple[Indexer, Retrieval]:
        """Build the RAG indexer and retrieval, on the embedding model the settings ask for.

        Both share one collection, and so one client: an embedded Qdrant locks
        its directory for a single client, so a second one on the same location
        could not open it.

        Raises:
            ModelError: If no provider serves the embedding model the settings
                ask for, which indexing or asking anything would need.

        """
        model = self._settings.llm.embedding_model or DEFAULT_EMBEDDING_MODEL
        provider = self._provider_manager.get_provider(model=model)
        if provider is None:
            msg = f"the RAG needs the embedding model '{model}', which no provider serves"
            raise ModelError(msg)
        indexation = Indexation(provider, model)
        collection = Collection.open(self._settings.db.rag.location)
        return Indexer(indexation, collection), Retrieval(indexation, collection, provider)

    def _build_web_tools(self) -> list[Tool]:
        """Build the web search and fetch tools, which need an embedding model."""
        embedding_model = self._settings.llm.embedding_model
        if embedding_model is None:
            return []
        provider = self._provider_manager.get_provider(embedding_model)
        if provider is None:
            err = f"the web tools need the embedding model '{embedding_model}', which no provider serves"
            raise ModelError(err)
        embedder = Embedder(provider, embedding_model)
        return web_tools(self._settings.web, embedder)

    def _launch_command(self, content: str) -> None:
        """Parse `content` as a `/name args...` command and run it, reporting unknown commands to the UI."""
        parts = content.removeprefix("/").split()
        if not parts:
            return
        name, *args = parts
        try:
            self._ui.send_input(Role.USER, content, command=True)
            self._command_registry.execute_command(name, *args)
        except UnknownCommandError as err:
            self._ui.send_system_message(str(err))

    def _launch_conv_chat(self, content: str) -> None:
        """Send `content` through the conversation, reporting any provider error instead of raising."""
        if self.conversation.uuid is None:
            self.conversation.add_conv_to_db(None, content)
        try:
            self.conversation.chat(content, self._ui)
        except SensaiError as err:
            print(str(err))
            # replace with log

    @staticmethod
    def _is_quit(event: ui.Event | None) -> bool:
        return event is not None and event.type is ui.EventType.Command and event.content == "/q"
