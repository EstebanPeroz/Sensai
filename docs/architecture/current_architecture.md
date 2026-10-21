# Current architecture

State of the code on `main` @ `dadf395`.

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

    CONV <--> LLM
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

Semantic cache: before calling the model, `Conversation` looks the query up in the cache, scoped by the model in use,
the active persona and the embedding model, and compared with the previous user messages as context. On a hit, the
cached answer is sent to the TUI and saved like any other; on a miss, the complete answer is stored with a TTL. Without
an embedding model or a reachable Redis server, `Core` builds no cache and every query goes to the model.
