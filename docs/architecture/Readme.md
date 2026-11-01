## Layout

Sensai is a uv workspace of three packages: the server holds the logic, the clients only draw it, and the protocol
package is what both import to talk to each other. How they work together is in
[current_architecture.md](current_architecture.md).

```text
Sensai/
├── pyproject.toml                 # workspace members, dev dependencies, ruff and pytest config
├── compose.yaml                   # Qdrant and Redis
├── docs/
├── tests/
│   ├── protocol/
│   ├── server/                    # one folder per server package below
│   └── client/tui/
└── packages/
    ├── protocol/src/sensai_protocol/
    │   ├── client_messages.py     # login, chat, command, cancel, complete, the *_answer messages
    │   ├── server_messages.py     # session, ai_chunk, system, error, history, the questions to the client
    │   ├── client_commands.py     # read_file, read_binary_file, write_file: params and results
    │   ├── dashboard.py           # DashboardReport, sent by /dashboard
    │   └── parsing.py             # parse_client_message, parse_server_message
    │
    ├── server/
    │   ├── settings.toml(.template)
    │   └── src/sensai_server/
    │       ├── __init__.py        # main: load the settings, build the app, run uvicorn
    │       ├── parsing.py         # --config, --port
    │       ├── events.py          # EventBus, the Observer side
    │       ├── app/               # composition root and websocket endpoint
    │       │   ├── app.py         # create_app, ws_endpoint, lifespan
    │       │   ├── _handlers.py   # one handler per client message
    │       │   ├── _state.py      # AppState: every shared service
    │       │   └── outbound.py    # sends server messages, awaits the client's answers
    │       ├── session/           # Session (one per client) and SessionRegistry
    │       ├── chat/              # run a chat through the pipeline, ReAct loop, context usage
    │       ├── pipeline/          # Chain of Responsibility
    │       │   ├── chain.py       # ChatContext, the stage contract
    │       │   ├── chat_pipeline.py
    │       │   └── modules/       # guardrails/, cache, compression, persona, rag, model
    │       ├── commands/          # one Command per slash command, CommandRegistry
    │       ├── llm/               # providers (Ollama), embedder, personas, streamed responses
    │       ├── tools/             # Tool, ToolManager, per-session ToolState, web/, mcp/, ask_user
    │       ├── rag/               # sources, chunking, Qdrant collections, search
    │       ├── cache/             # semantic cache on Redis, one SessionCache per session
    │       ├── memory/            # conversation, history, summaries, persistent_db/ (SQLAlchemy)
    │       ├── monitoring/        # Interaction, recorder, logger, context budget, dashboard metrics
    │       └── config/            # Settings, shipped personas
    │
    └── client/tui/src/sensai_tui/
        ├── __init__.py            # main: connect, then run the chat app
        ├── parsing.py             # --model, --port
        ├── connection.py          # the websocket: send client messages, yield server messages
        ├── completion.py
        ├── client_commands/       # what the server may ask the client to run on its machine
        └── ui/                    # Textual app, dialogs, dashboard screen
```

## Design patterns, by package

- **`sensai-protocol` — shared contract.** Each message is a frozen pydantic model with a literal `type`, and each
  direction is a discriminated union, so one call parses any frame and rejects unknown fields. Neither side defines
  a message the other does not know.

- **`app/` — composition root.** `create_app` builds every service once and keeps them in `AppState`; nothing else
  constructs a shared service. `ws_endpoint` only parses and dispatches: each client message has its handler in
  `_handlers.py`.

- **`session/` — one session per client.** The shared services hold no per-client state; a `Session` carries it
  (conversation, model, persona, effort, `ToolState`, `SessionCache`) and runs at most one task at a time, so a chat,
  a command and a cancel never race.

- **`app/outbound.py` — request/response over the socket.** The server-side questions (`confirm_tool`, `ask_user`,
  `ask_compression`, `request_command`) carry a `request_id`; `Outbound` awaits the future the matching answer
  resolves. Code deep in a tool or a command just awaits `out.ask_confirmation(...)` without knowing about the
  socket.

- **`pipeline/` — Chain of Responsibility.** `ChatPipeline` links stages around the model stage: guard → cache →
  compression → persona → RAG → model. Each stage can transform the `ChatContext`, answer itself (a guard refusal, a
  cache hit) or pass on. The stages are shared; per-session data comes in the context.

- **`llm/` — Adapter.** `ProviderManager` finds the provider serving a model; `ollama.py` is the only concrete one.
  Its calls are blocking and are streamed into the event loop through `threaded.py`.

- **`tools/` — Strategy + Registry.** Every tool (`web_search`, `web_fetch`, `ask_user`, each MCP tool) implements
  the `Tool` contract and is looked up by name through `ToolManager`, which also enforces confirmation. MCP tools
  go through `MCPHost`, one session per MCP server on its own thread.

- **`commands/` — Command.** Each slash command is its own object with `execute`, `help`, `usage` and
  completions, registered in the `CommandRegistry`. Commands act on the user's session; tools are what the model
  calls during reasoning.

- **`memory/persistent_db/` — Repository.** `PersistentDatabase` is the storage interface (branches, messages,
  personas, summaries, interactions); the SQLAlchemy implementation is the only one.

- **`monitoring/` — Observer.** Every model call becomes an `Interaction` emitted on the app-wide `EventBus`.
  `ContextBudget` and `InteractionLogger` subscribe to it; the code making the call never knows who listens.

- **`client_commands/` — the client's side of the contract.** A handler per `ClientCommand`, run in the client's
  working directory. The server never opens a path itself.

## Configuration surface

Everything that should change without touching code is in the server's settings file (`--config`): providers,
embedding model, databases, cache, context budget, disabled and confirmed tools, security personas, logging, MCP
servers. The client only takes `--model` and `--port`.
