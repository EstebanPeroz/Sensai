# Architecture

```mermaid
flowchart TB
    MAIN["main.py"]
    CONFIG["config<br/>Settings"]
    CORE["Core"]

    subgraph UI["ui"]
        UIA{{"UIAdapter"}}
        TUI["Textual TUI"]
    end

    CMDS["commands<br/>/help, /model, /persona"]
    CONV["Conversation<br/>+ History"]
    REACT["ReAct<br/>think / act / observe"]
    TOOLS["ToolManager<br/>web_search, web_fetch, MCP tools, ask_user"]

    subgraph LLM["llm"]
        PM["ProviderManager"]
        OLLAMA["OllamaAdapter"]
        PERSONA["PersonaManager"]
    end

    DB{{"PersistentDatabase<br/>(SqlAlchemy)"}}
    CACHE{{"SemanticCacheRepository<br/>(RedisSemanticCache)"}}
    BUS["EventBus"]
    LOGGER["InteractionLogger"]
    INTERACTIONS{{"InteractionRepository<br/>(SqlInteractionRepository)"}}

    OLLAMAAPI[("Ollama API")]
    SQLITE[("SQLite")]
    REDIS[("Redis")]

    MAIN --> CONFIG
    MAIN --> TUI
    MAIN --> CORE

    CORE <--> UIA
    CORE --> CMDS
    CORE --> CONV

    TUI -.->|implements| UIA

    CMDS <--> LLM

    CONV --> REACT
    REACT <--> LLM
    REACT --> TOOLS
    TOOLS -.->|asks confirmation, ask_user| UIA
    CONV <--> DB
    CONV <-->|lookup / store| CACHE
    CONV <-->|streams responses| UIA

    PM <--> OLLAMA
    LLM <--> OLLAMAAPI
    PERSONA <--> DB
    DB <--> SQLITE
    CACHE <--> REDIS
    CACHE -->|embeds queries| LLM

    CORE --> BUS
    CONV -->|emits each model call| BUS
    BUS --> LOGGER
    LOGGER --> INTERACTIONS
    INTERACTIONS <--> SQLITE
```

Message flow: the TUI pushes the user input into an event queue, and `Core` reads it and hands it to `Conversation`.
`Conversation` saves the message to the DB, then calls Ollama in streaming mode. The response chunks are sent back to
the TUI as they arrive.

Before a gated tool runs, `ToolManager` asks the user through `UIAdapter.ask_confirmation`, which opens a modal
dialog and blocks the streaming thread until it closes. `ask_user` asks its question the same way (see `technical-notes.md`, "Human in the loop").

Semantic cache: before calling the model, `Conversation` looks the query up in the cache, scoped by the model in use,
the active persona and the embedding model, and compared with the previous user messages as context. On a hit, the
cached answer is sent to the TUI and saved like any other; on a miss, the complete answer is stored with a TTL, unless
it was built with tools, whose results (web pages, search results) may soon be outdated. Without an embedding model or
a reachable Redis server, `Core` builds no cache and every query goes to the model.

Logging: every call to the model made by the ReAct loop of `Conversation` or by the RAG answer of `/ask` is recorded as
an `Interaction` (request, response, tool calls, latency, token counts, status) and published on the `EventBus`.
`InteractionLogger` subscribes to it and stores each one through `InteractionRepository`, in the `interaction` table of
the persistent database. The `[logging]` settings disable it, or drop the content and keep only the metrics. A storage
failure is reported to the UI without breaking the answer.

## Server monitoring

The server (`packages/server`) records the model calls of every session on a single, app-wide bus.

```mermaid
flowchart LR
    subgraph APP["create_app → AppState"]
        BUS["EventBus<br/>(events)"]
        LOGGER["InteractionLogger<br/>(interaction_logger)"]
    end

    RUNCHAT["run_chat<br/>ChatContext.on_interaction"]
    REACT["ReAct._request<br/>InteractionRecorder"]
    PROVIDER[("Ollama API")]
    REPO{{"InteractionRepository<br/>(db.interactions)"}}
    DASH["/dashboard<br/>metrics.build_report"]
    SQLITE[("SQLite")]

    RUNCHAT -->|builds the context| REACT
    REACT <-->|streams| PROVIDER
    REACT -->|"Interaction (OK / ERROR / INTERRUPTED)"| RUNCHAT
    RUNCHAT -->|tags branch_id + persona, emit| BUS
    BUS --> LOGGER
    LOGGER -->|"asyncio.to_thread(add)"| REPO
    DASH -->|"asyncio.to_thread(get_all)"| REPO
    REPO <--> SQLITE
```

`AppState`: `create_app` builds one `EventBus` and one `InteractionLogger` for the whole server and keeps both in
`AppState` (`events`, `interaction_logger`), next to `db` and `pipeline`. The logger subscribes to `Interaction` at
startup, unless `[logging] enabled = false`. Every session emits on that same bus.

Recording: `ReAct._request` is the only place the server calls the model, so each `provider.chat` stream (think, act,
answer…) gives one `Interaction`. An `InteractionRecorder` accumulates the chunks, the first token time and the last
`ChatMetrics`. In the `finally` of the generator, the record is passed once to `ChatContext.on_interaction` with its
status: `OK`, `ERROR` (an error chunk or a `SensaiError`) or `INTERRUPTED` (closed or cancelled: `/cancel`, a new
message, logout, disconnection). The error or cancellation still propagates. `ReAct` knows neither the session nor the
bus: `run_chat` builds `on_interaction`, which tags the record with the branch and the persona and emits it, catching
and logging any subscriber failure so the answer is never broken. An answer given without the model is not recorded: a
prompt the guard rejects, or a semantic cache hit (`CacheStage` sits before the model stage). On a miss, the call is
recorded as usual and the cache stores the answer.

Async logger: `emit` and `on_interaction` are synchronous, because they run while a generator may be closing.
`InteractionLogger.log` therefore never writes inline: it strips the content when `log_content = false`, then schedules
`repository.add` in a thread as a background task and keeps it until done. A failed write is logged on the server
only. At shutdown, `_lifespan` closes the sessions first (their running calls are emitted as `INTERRUPTED`), then
awaits `drain()` so the pending writes are stored.

`/dashboard [24h|7d|30d]` reads every logged call of the period and its baseline, across all sessions, builds the
report with `monitoring/metrics.py`, and sends it to the client as a `DashboardReport`.
