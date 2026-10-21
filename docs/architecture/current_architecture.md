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
    TOOLS["ToolManager<br/>web_search, web_fetch, ask_user"]

    subgraph LLM["llm"]
        PM["ProviderManager"]
        OLLAMA["OllamaAdapter"]
        PERSONA["PersonaManager"]
    end

    DB{{"PersistentDatabase<br/>(SqlAlchemy)"}}

    OLLAMAAPI[("Ollama API")]
    SQLITE[("SQLite")]

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
    CONV <-->|streams responses| UIA

    PM <--> OLLAMA
    LLM <--> OLLAMAAPI
    PERSONA <--> DB
    DB <--> SQLITE
```

Message flow: the TUI pushes the user input into an event queue, and `Core` reads it and hands it to `Conversation`.
`Conversation` saves the message to the DB, then calls Ollama in streaming mode. The response chunks are sent back to
the TUI as they arrive.

Before a gated tool runs, `ToolManager` asks the user through `UIAdapter.ask_confirmation`, which opens a modal
dialog and blocks the streaming thread until it closes. `ask_user` asks its question the same way (see `technical-notes.md`, "Human in the loop").
