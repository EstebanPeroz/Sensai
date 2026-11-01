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

## Server: MCP host

The server (`packages/server`) offers the tools of external MCP servers next to its built-in tools. `MCPHost`
(`sensai_server/tools/mcp/host.py`) keeps one session open to every enabled `[mcp.<name>]` server, from startup to
shutdown, on **its own anyio event loop in a background thread**, beside uvicorn's loop:

```mermaid
flowchart LR
    subgraph UVICORN["uvicorn loop"]
        REACT["ReAct"] --> TM["ToolManager"] --> TOOL["MCPTool.run"]
    end

    subgraph HOSTTHREAD["MCPHost thread (own anyio loop)"]
        SERVE["_serve<br/>one task group"]
        S1["MCPServerSession(fs)"]
        S2["MCPServerSession(remote)"]
        SERVE --> S1
        SERVE --> S2
    end

    TOOL -->|"wrap_future(host.call)"| S1
    S1 <-->|stdio| FS[("subprocess")]
    S2 <-->|streamable HTTP| REMOTE[("remote server")]
```

- **Creator: `build_tools`.** `create_app` runs before uvicorn starts its loop, so `build_tools` creates the host,
  starts it (servers open concurrently, each listing its tools under `startup_timeout`), and appends one `MCPTool`
  per tool, named `<server>__<tool>`. A server that fails to start is logged and its tools are left out; the app
  still starts. `build_tools` returns the host with the tools, and `AppState.mcp` keeps it.
- **Calls.** `MCPTool.run` awaits `asyncio.wrap_future(host.call(...))`, which schedules `session.call_tool` on the
  host's loop through its portal, so uvicorn's loop never blocks. Cancelling the chat cancels the asyncio future, and
  through it the call on the host's loop. One `ClientSession` per server serves every client session at once.
- **Disconnect.** A dropped subprocess or connection is logged once; every later call to that server returns an
  `error: MCP server '<name>' disconnected…` message to the model. There is no reconnection.
- **Closer: `_lifespan`.** On shutdown, after `registry.close_all()` has cancelled the running tool calls, it runs
  `asyncio.to_thread(state.mcp.close)`, which closes every session, reaps every subprocess and stops the thread.
  `close` is idempotent: `create_app` also calls it when a later startup step fails, and `main` in a `finally` around
  `uvicorn.run`, so no thread or subprocess outlives the process.
