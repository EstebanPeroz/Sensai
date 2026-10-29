## Target layout

```text
sensai/
├── README.md
├── requirements.txt
├── config/
│   ├── settings.toml              # model, active handlers, active tools
│   └──  guardrails.toml            # filtering rules
├── docs/
│   ├── architecture/
│   └── user_stories/
├── tests/
└── sensai/
    ├── __main__.py                # `python -m sensai`, delegates to main.py
    ├── main.py                    # composition root: builds and injects everything
    ├── parsing.py                 # command line arguments
    │
    ├── core/                      # Mediator
    │   ├── core.py                # receives every module as a dependency
    │   └── events.py              # EventBus, emitter for the Observer side
    │
    ├── ui/
    │   ├── adapter.py                # interface
    │   ├── tui/                     # Textual chat app, and the dashboard screen
    │   └── web/
    │
    ├── pipeline/                  # Chain of Responsibility
    │   ├── chain.py               # generic Chain, instantiated per direction
    │   └── modules/
    │       ├── model.py                # module interface
    │       ├── precompute/             # inbound chain
    │       │   ├── pii_masking.py
    │       │   ├── compression.py      # summarises the oldest messages near the token budget
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
    │   ├── structured_output.py    # decorator around the adapter
    │   └── thinking/
    │       └── react.py            # ReAct reasoning loop: think, act, observe
    │
    ├── tools/                     # Strategy + Registry
    │   ├── __init__.py             # defines the entry points
    │   ├── tool.py                 # tool contract: name, description, parameters, run
    │   ├── tool_manager.py         # looks tools up by name and runs them
    │   ├── _mcp.py                 # Fetch
    │   ├── _human_in_loop.py
    │   ├── _web_search.py
    │   ├── _file_access.py
    │   └── _rag_retrieval.py
    │
    ├── commands/                  # Command
    │   ├── __init__.py             # execute()
    │   ├── _model.py                # choose a model
    │   ├── _effort.py               # thinking effort of the reasoning loop
    │   ├── _branch.py               # branch off the current session
    │   ├── _compress.py             # summarise the oldest messages now
    │   ├── _export.py               # export a conversation
    │   ├── _dashboard.py            # monitoring dashboard of the logged calls
    │   └── _schedule.py             # schedule a task (time / action)
    │
    ├── memory/                    # Repository + Memento
    │   ├── history_repository.py   # conversation history interface
    │   ├── history.py              # DB-backed conversation history
    │   ├── summary.py              # summary of the oldest messages of a branch, and its storage interface
    │   ├── profile.py
    │   ├── vector_index.py
    │   ├── ingest.py                # chunking + embeddings for RAG
    │   └── memento.py               # snapshots backing /branch and /export
    │
    ├── semantic/                  # Repository
    │   ├── cache_repository.py     # semantic cache interface and its scope
    │   ├── cache.py                # Redis-backed semantic cache
    │   ├── cache_key.py            # what a query is cached under, and the cached response
    │   └── connection.py           # connects to Redis, or no cache if unreachable
    │
    └── monitoring/                # Observer + Repository
        ├── interaction.py          # record of one call to the model, the event emitted on the bus
        ├── recorder.py             # builds that record from the chunks of a call
        ├── interaction_repository.py # storage interface of the logged calls
        ├── logger.py               # bus subscriber storing every call, with or without its content
        ├── budget.py               # bus subscriber measuring the context the conversation takes
        └── metrics.py              # dashboard report: percentiles, time series, drift detection
```

## Design patterns, by package

- **`core/` — Mediator.** `core.py` is the hub every other module is wired
  into; it doesn't own logic itself, it coordinates: it builds the reasoning
  loop and its tools, and hands them to the conversation. `events.py` is an
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
  `thinking/react.py` implements the ReAct-style reasoning loop on top of
  the adapter (see `technical-notes.md`).

- **`tools/` — Strategy + Registry.** Every tool (`_mcp.py`,
  `_human_in_loop.py`, `_web_search.py`, `_file_access.py`,
  `_rag_retrieval.py`) implements a common tool interface (Strategy) and is
  looked up by name through `tool_manager.py`, which is what the reasoning
  loop and `tool_calls` payloads resolve against.

- **`commands/` — Command.** Each user-facing slash-style action
  (`_model`, `_effort`, `_branch`, `_compress`, `_export`, `_schedule`) is its own
  encapsulated command object with an `execute()` entry point, dispatched
  through `commands/__init__.py`. This is distinct from `tools/`: commands
  are user/session-lifecycle actions, tools are things the *model* invokes
  during reasoning.

- **`memory/` — Repository + Memento.** Each repository has its own
  storage interface (`history_repository.py` for the conversation history),
  so the history, user profile and RAG stores stay independent;
  `history.py` (conversation history) and `profile.py` (user profile) are
  concrete repositories. `vector_index.py` + `ingest.py` back
  the RAG feature (chunking and embedding on ingest, similarity search at
  query time). `memento.py` captures session snapshots, which is what
  `/branch` (fork a session) and `/export` (serialize a session) operate on.

- **`semantic/` — Repository.** Everything the semantic cache needs is grouped
  here, apart from the conversation and the core. `cache.py` stores past
  answers in Redis with their query and context embeddings and a TTL, scoped
  by model, persona and embedding model (`cache_repository.py` is its
  interface). `cache_key.py` builds what a query is cached under from the
  history, and `connection.py` gives the core no cache when Redis is
  unreachable. The conversation looks the query up before calling the model
  and answers from the cache on a hit.

- **`monitoring/` — Observer + Repository.** Every call to the model, whatever
  its outcome (done, failed, interrupted), becomes an `Interaction`: what was
  sent, what came back, the tools called, the latency and the token counts.
  The code that calls the model (the ReAct loop, the RAG answer) records it
  with `recorder.py` and hands it to a callback; the core publishes it on the
  `EventBus`, so the callers never know whether, or where, it is stored.
  `logger.py` is the subscriber that writes it through `InteractionRepository`
  (`interaction_repository.py`), which the persistent database implements
  in its own `interaction` table; the `[logging]` settings turn it off, or keep
  only the metrics without what the user and the model wrote. A failure to
  store a call is reported to the UI and never breaks the answer.
  `/dashboard [24h|7d|30d]` reads the calls of the period back from the
  repository, without their content, and `metrics.py` turns them into a
  `Report`: call count, response time p50/p95 and first token p50 (successful
  calls only), error and interruption rates, tokens per second, a time series
  per hour, 6 hours or day, the figures per model and persona, the latest
  errors, and the response time drifts against the 7 previous periods. The UI
  shows it with `show_dashboard` (a Textual screen over the chat).
  `budget.py` is another subscriber: after each chat call, it takes the real
  token count of the call (prompt and answer) as the context the conversation
  takes, against the `[context]` budget, or the context length of the model
  without one. The UI shows it on the input box, in red past the warning
  threshold. Past it, the next message first asks the user, in a dialog
  (`ask_compression`), whether to compress the conversation and go on, or to
  start a new conversation with that message. To go on, the compression stage
  of the pipeline asks the model to
  summarise all but the last messages, keeping figures, names and decisions;
  the summary is stored per branch with the number of messages it stands for,
  sent in their place from then on, and extended by the next compression.
  `/compress` runs it on demand. The messages themselves stay in the branch.

## Configuration surface

`config/` centralizes everything that should be changeable without touching
code — directly serving the brief's "model selectable via CLI argument or
configuration file, without modifying the code" requirement, and extending
the same idea to personas, guardrail rules, file-access permissions, and
per-subject/project system prompts (the "Other: Extension system prompt via
Sujet/Projet" line in `info/user_group.md`).
