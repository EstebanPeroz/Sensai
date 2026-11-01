# Implemented features

Features of [the catalogue](features_catalog.md) implemented in Sensai, with how to use them. The user stories and
user scenarios behind each one are in [user_stories/analyst.md](user_stories/analyst.md).

Status: **done** covers the whole catalogue entry, **partial** covers part of it, the missing part being named.

## Summary

| ID | Feature | Status | How to use it |
|---|---|---|---|
| B1 | Functional CLI chatbot | done | `make run`, `--model`, `/model` |
| M1 | Session Persistence & Profiles | partial | `/branch list`, `/branch set <uuid>` |
| M2 | Token Budgeting & Semantic Compression | done | `[context]` settings, `/compress` |
| T1 | MCP | done | `[mcp.<name>]` tables in `settings.toml` |
| T3 | Web Search | done | `web_search` and `web_fetch` tools |
| R1 | Basic RAG | done | `/index <path\|url>`, `/ask <question>`, plain chat |
| R2 | Advanced RAG | partial | automatic |
| A1 | Reasoning Loops (ReAct) | done | `/effort off\|low\|medium\|high` |
| A3 | Human-in-the-Loop | done | `[features].confirm`, `ask_user` tool |
| A5 | Persona | done | `/persona`, `[personas]` settings |
| A6 | Semantic Cache | done | Redis + `[cache]` settings |
| EV2 | Content & Privacy Guardrails | partial | automatic, security persona |
| EV3 | Logging & Monitoring Dashboard | done | `[logging]` settings, `/dashboard [24h\|7d\|30d]` |
| X2 | Branching — Resume & Interrupt | done | `/branch fork`, Escape or Ctrl+C mid-answer |
| X3 | Export & Publication | partial | `/export current\|<uuid>`, `/import <path>` |
| X4 | Questions & Forms | partial | `ask_user` tool |

Not implemented: M3, M4, T2, T4, T5, A2, A4, A7, EV1, EV4, X1, X5.

## Traceability

From each feature to the user stories it answers, the scenario that demonstrates it, the code that implements it,
and the tests that verify its acceptance criteria. The stories and scenarios are in
[user_stories/analyst.md](user_stories/analyst.md).

| ID | User stories | Scenario | Code | Tests |
|---|---|---|---|---|
| B1 | US-59, US-60, US-61 | 9 | `llm/ollama.py`, `history/conversation.py`, `parsing.py`, `commands/model.py` | `test_ollama.py`, `test_conversation.py`, `test_parsing.py`, `commands/test_commands_model.py` |
| M1 | US-01, US-62 | 6 | `memory/persistent_db/`, `commands/branch.py` | `persistent_db/test_sql_alchemy.py`, `commands/test_commands_branch.py` |
| M2 | US-03, US-04, US-63 | 6 | `monitoring/budget.py`, `pipeline/modules/compression.py`, `commands/compress.py` | `monitoring/test_budget.py`, `pipeline/test_compression.py`, `persistent_db/test_summaries.py`, `commands/test_commands_compress.py` |
| T1 | US-05, US-06, US-58 | 4 | `mcp/` | `mcp/test_client.py`, `mcp/test_config.py` |
| T3 | US-07, US-08 | 1, 3 | `tools/web/` | `test_web_search.py` |
| R1 | US-13, US-14 | 5 | `rag/`, `pipeline/modules/rag.py`, `commands/index.py`, `commands/ask.py` | `rag/`, `commands/test_commands_index.py`, `commands/test_commands_ask.py` |
| R2 | US-16, US-64 | 5 | `rag/retrievial_rag.py`, `rag/indexation_rag.py` | `rag/test_retrieval.py`, `rag/test_indexation.py` |
| A1 | US-17, US-18, US-47, US-48 | 1 | `llm/thinking/react.py`, `commands/effort.py` | `test_react.py`, `commands/test_commands_effort.py` |
| A3 | US-19, US-54, US-55, US-56 | 3 | `tools/tool_manager.py`, `commands/tools.py` | `test_tool_manager.py`, `commands/test_commands_tools.py` |
| X4 | US-20, US-57 | 3 | `tools/ask_user.py` | `test_ask_user.py` |
| A5 | US-23, US-24, US-65 | 2, 7 | `llm/persona.py`, `llm/persona_manager.py`, `commands/persona.py` | `test_persona.py` |
| A6 | US-49, US-50, US-51 | 2 | `semantic/` | `test_semantic_cache.py` |
| EV2 | US-66, US-67 | 7 | `pipeline/modules/guardrails/`, `config/personas/security.toml` | `server/test_server_bad_words.py` |
| EV3 | US-52, US-53, US-68 | 8 | `monitoring/`, `commands/dashboard.py` | `monitoring/`, `test_conversation_logging.py`, `persistent_db/test_interactions.py`, `commands/test_commands_dashboard.py` |
| X2 | US-37, US-69, US-70 | 6 | `commands/branch.py`, `ui/` | `commands/test_commands_branch.py`, `client/tui/test_tui_app.py` |
| X3 | US-39, US-71 | 6 | `commands/data_transfer/` | `server/test_server_export.py`, `server/test_server_import.py` |

Code paths are relative to the `sensai` package, test paths to `tests/`. The tests mock Ollama, Redis and the vector
database where needed; `make tests` runs them all.

## Details

### B1 — Functional CLI chatbot

- Answers are streamed token by token into the chat.
- Every turn is sent to the model with the previous ones of the conversation.
- Errors are reported instead of crashing: unreachable Ollama or missing model at start (one-line error, exit code
  1, no traceback), unknown model on `/model`, unknown command, empty input ignored. With no Ollama and an empty
  stdin, the program exits cleanly (checked by the CI).
- The model is chosen with `--model <name>` (or `make run MODEL=<name>`), switched at runtime with `/model <name>`,
  and `/model` alone lists the models Ollama serves. The settings file is chosen with `--config <path>`.
- `/help` lists the commands, and commands and their arguments are completed with Tab.

### M1 — Session Persistence & Profiles (partial)

- Every conversation (branch) is stored in SQLite (`db.persistent.url`) with its messages and its model.
- `/branch list` lists the stored conversations, `/branch set <uuid>` reloads one, history included, in a later
  session.
- Personas are stored in the same database and survive restarts.
- Missing: a user profile (preferences, custom instructions) injected automatically at start. The closest is a
  persona carrying those instructions.

### M2 — Token Budgeting & Semantic Compression

- The tokens the conversation takes are counted after every call (from the counts Ollama reports) against the budget
  of `[context].budget`, or the context length of the model without one. The budget is also sent to Ollama as its
  context window.
- The usage is shown at the top left of the input box as `context: used / budget (share)`, in red past
  `warning_threshold`.
- Past the threshold, the next message opens a dialog: compress the conversation, or start a new one.
- Compression summarises every message but the `keep_recent` last ones with the current model. The messages stay in
  the database; only the summary, sent in their place, is stored, and the next compression extends it.
- `/compress` compresses at any time and tells how many messages were summarised and the tokens saved.

### T1 — MCP

- Any number of MCP servers, one `[mcp.<name>]` table each, local (`stdio`) or remote (streamable `http`, with
  `headers` for tokens). Their tools are offered to the model as `<server>__<tool>`.
- Servers start concurrently; one that fails or times out is skipped with a message, the others still start.
- One session per server is kept open until Sensai quits, so a server keeps its state between calls.
- A failing call is sent back to the model as an `error: ...` result.
- Tested with our own server (`tests/mcp/fixtures/tiny_server.py`, stdio and http) and the official filesystem
  server. Details in the README.

### T3 — Web Search

- `web_search` returns the title, URL and snippet of the top results; `web_fetch` returns the main text of a page.
- Each tool has fallback backends: Ollama's web API (with `[web].api_key`), then DuckDuckGo (`ddgs`) for search and
  a direct HTTP fetch for pages.
- Results are cleaned and cut to a token limit before reaching the model.
- When every backend fails, the model is told the information could not be verified online, instead of guessing.

### R1 — Basic RAG

- `/index <path|url> ...` ingests text, Markdown, HTML and PDF files and web pages: cleaning, chunking, embedding
  (`[llm].embedding_model`), storage in Qdrant (Docker, or embedded with a directory as `db.rag.location`).
- Re-indexing a source replaces its chunks, so answers follow the latest version.
- In the chat, the chunks closest to the message are added to the context, and the answer ends with
  `Sources: ...`. `/ask <question>` answers from the documents only.
- A failed search is reported, and the model answers without the documents.

### R2 — Advanced RAG (partial)

- Per-chunk relevance scoring: chunks under a minimum similarity are dropped before reaching the context, and the
  others come best first.
- Chunking adapted to the document: Markdown split on its headings, other text split where the meaning changes
  (semantic chunking on sentence embeddings).
- Missing: reranking and metadata filtering.

### A1 — Reasoning Loops (ReAct)

- Think, act (call a tool), observe, until the model has an answer or reaches the step limit. The reasoning is shown
  in the chat, and each tool call as `-> Using <tool>...`.
- `/effort off|low|medium|high` allows 0, 1, 3 or 5 steps; `off` answers in a single request without tools. The
  level is shown at the bottom right of the input box.

### A3 — Human-in-the-Loop

- Tools listed in `[features].confirm` open a dialog before they run, with the tool name and its arguments: Allow
  (`y`), Always this session (`a`), Deny (`n` or Escape).
- A denied call is reported to the model, which answers without it.
- `/tools list` shows which tools ask for confirmation; `/tools enable|disable <name>` toggles a tool.

### A5 — Persona

- A persona (name, description, prompt) is a TOML file; `[personas]` loads files at start, `/persona create <path>`
  adds one at runtime.
- `/persona list | info <name> | load <name> | update <name> <field>=<value> ...`. Loading a persona keeps the
  conversation history.
- Shipped personas: `analyst`, and `security` (see EV2).

### A6 — Semantic Cache

- Answers are stored in Redis with their query embedding and a TTL (`[cache].ttl`), and served instantly to a query
  whose similarity passes `[cache].similarity_threshold`.
- A cached answer is only reused for the same model, persona and embedding model, and a similar preceding
  discussion.
- Answers built with tools are not cached, their sources being likely to change.
- Without Redis or an embedding model, Sensai runs without the cache.

### EV2 — Content & Privacy Guardrails (partial)

- Input guard: a message containing an abusive word is answered with a warning and never reaches the model.
- The `security` persona makes the model refuse out-of-scope requests, attempts to change or reveal its
  instructions, and content holding unmasked personal data.
- Missing: PII detection and masking in code, output filtering.

### EV3 — Logging & Monitoring Dashboard

- Every call to the model is stored in the `interaction` table: request, response, tool calls, latency, token
  counts, status (ok, error, interrupted), model, persona.
- `[logging].enabled = false` turns it off; `log_content = false` keeps only the metrics.
- `/dashboard [24h|7d|30d]` shows call volume over time, response time percentiles, throughput, token totals, error
  and interruption rates per model and persona, and flags a drift when the median response time rises over the
  previous periods.

### X2 — Branching — Resume & Interrupt

- Resume: `/branch fork` copies the current conversation into a new branch and switches to it, the original staying
  untouched. `/branch set`, `rename`, `remove` manage the branches.
- Interrupt: Escape or Ctrl+C stops the running answer, and the next message redirects the conversation.
- Missing: editing a past message to fork from it.

### X3 — Export & Publication (partial)

- `/export current|<uuid>` writes a branch to `<uuid>.json`; `/import <path>` recreates a branch from such a file.
- Missing: Markdown/PDF export and public links.

### X4 — Questions & Forms (partial)

- The model can call the `ask_user` tool when a detail is missing: the question opens a dialog, the answer goes back
  to the model, and both are shown in the chat. Escape skips the question.
- Missing: choice lists and multi-field forms.
