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
