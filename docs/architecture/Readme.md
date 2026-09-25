## Target layout

```text
sensai/
├── README.md
├── requirements.txt
├── config/
│   ├── settings.toml              # model, active handlers, active tools
│   ├── personas/                  # one .toml per persona
│   ├── subjects/                  # prompts per subject/project
│   ├── guardrails.toml            # filtering rules
│   └── permissions.toml           # allowed paths for the file_access tool
├── docs/
│   ├── architecture/
│   └── user_stories/
├── tests/
└── sensai/
    ├── __main__.py                # composition root: builds and injects everything
    │
    ├── core/                      # Mediator
    │   ├── core.py                # receives every module as a dependency
    │   ├── reasoning_loop.py      # drives the chain of tool calls
    │   └── events.py              # EventBus, emitter for the Observer side
    │
    ├── ux/
    │   ├── model.py                # interface
    │   ├── tui/
    │   ├── api/
    │   └── web/
    │
    ├── pipeline/                  # Chain of Responsibility
    │   ├── chain.py               # generic Chain, instantiated per direction
    │   └── modules/
    │       ├── model.py                # module interface
    │       ├── precompute/             # inbound chain
    │       │   ├── pii_masking.py
    │       │   ├── compression.py
    │       │   └── prompt_optimization.py
    │       └── guardrails/             # outbound chain
    │           ├── hallucination.py
    │           ├── content_filter.py
    │           └── injection_defense.py
    │
    ├── llm/                       # Adapter
    │   ├── adapter.py              # LLMAdapter interface
    │   ├── ollama.py               # HTTP client, streaming
    │   ├── persona.py
    │   └── structured_output.py    # decorator around the adapter
    │
    ├── tools/                     # Strategy + Registry
    │   ├── __init__.py             # defines the entry points
    │   ├── registry.py
    │   ├── _mcp.py                 # Fetch
    │   ├── _human_in_loop.py
    │   ├── _web_search.py
    │   ├── _file_access.py
    │   └── _rag_retrieval.py
    │
    ├── commands/                  # Command
    │   ├── __init__.py             # execute()
    │   ├── _model.py                # choose a model
    │   ├── _branch.py               # branch off the current session
    │   ├── _compress.py
    │   ├── _export.py               # export a conversation
    │   └── _schedule.py             # schedule a task (time / action)
    │
    └── memory/                    # Repository + Memento
        ├── repository.py           # interfaces
        ├── history.py              # DB-backed conversation history
        ├── profile.py
        ├── vector_index.py
        ├── ingest.py                # chunking + embeddings for RAG
        └── memento.py               # snapshots backing /branch and /export
```

## Design patterns, by package

- **`core/` — Mediator.** `core.py` is the hub every other module is wired
  into; it doesn't own logic itself, it coordinates. `reasoning_loop.py`
  implements the ReAct-style loop. `events.py` is an
  `EventBus` — the Observer half of the design — so the UX layer (and
  anything else) can react to streaming tokens, tool calls, etc. without the
  core depending on the UI.

- **`ux/` — pluggable front end.** `model.py` defines the interface a front
  end must implement; `tui/` and `web/` are two concrete implementations
  (CLI is the required base loop's front end, `web/` maps to the "Web UI"
  feature).

- **`pipeline/` — Chain of Responsibility.** One generic `Chain` class is
  instantiated twice: an inbound chain (`precompute/`: PII masking →
  compression → prompt optimization) that runs before the prompt reaches the
  model, and an outbound chain (`guardrails/`: hallucination check → content
  filter → injection defense) that runs on the model's response before it
  reaches the user. Each module is a link that can inspect, transform, or
  short-circuit the payload.

- **`llm/` — Adapter.** `adapter.py` is the `LLMAdapter` interface;
  `ollama.py` is the (only, per the brief's constraint) concrete adapter,
  talking to the Ollama HTTP API with streaming. `structured_output.py`
  wraps an adapter to enforce a schema on its output (decorator pattern).
  `persona.py` injects persona system prompts ahead of the adapter call.

- **`tools/` — Strategy + Registry.** Every tool (`_mcp.py`,
  `_human_in_loop.py`, `_web_search.py`, `_file_access.py`,
  `_rag_retrieval.py`) implements a common tool interface (Strategy) and is
  looked up by name through `registry.py`, which is what the reasoning loop
  and `tool_calls` payloads resolve against.

- **`commands/` — Command.** Each user-facing slash-style action
  (`_model`, `_branch`, `_compress`, `_export`, `_schedule`) is its own
  encapsulated command object with an `execute()` entry point, dispatched
  through `commands/__init__.py`. This is distinct from `tools/`: commands
  are user/session-lifecycle actions, tools are things the *model* invokes
  during reasoning.

- **`memory/` — Repository + Memento.** `repository.py` defines storage
  interfaces; `history.py` (conversation history) and `profile.py` (user
  profile) are concrete repositories. `vector_index.py` + `ingest.py` back
  the RAG feature (chunking and embedding on ingest, similarity search at
  query time). `memento.py` captures session snapshots, which is what
  `/branch` (fork a session) and `/export` (serialize a session) operate on.

## Configuration surface

`config/` centralizes everything that should be changeable without touching
code — directly serving the brief's "model selectable via CLI argument or
configuration file, without modifying the code" requirement, and extending
the same idea to personas, guardrail rules, file-access permissions, and
per-subject/project system prompts (the "Other: Extension system prompt via
Sujet/Projet" line in `info/user_group.md`).
