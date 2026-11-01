# Sensai

Local CLI assistant for analysts, talking to Ollama over its HTTP API.

## Features

Every feature, with how to use it, is described in [docs/features.md](docs/features.md); the user stories and user
scenarios behind them are in [docs/user_stories/analyst.md](docs/user_stories/analyst.md).

| Feature | Use |
|---|---|
| Streaming chat with history, model chosen with `--model` or `/model` | `make run` |
| Session persistence and branching (fork, resume, interrupt) | `/branch list\|set\|fork\|rename\|remove` |
| Token budget shown live, conversation compressed near the budget | `[context]` settings, `/compress` |
| ReAct reasoning loop with a thinking effort | `/effort off\|low\|medium\|high` |
| Web search and page fetch, with fallback backends | `web_search`, `web_fetch` tools |
| MCP servers, local (stdio) or remote (http) | `[mcp.<name>]` tables, see below |
| RAG over text, Markdown, HTML, PDF files and web pages, with sources | `/index`, `/ask`, plain chat |
| Human in the loop: tool confirmation and questions to the user | `[features].confirm`, `ask_user` tool |
| Personas switchable at runtime | `/persona list\|info\|load\|create\|update` |
| Semantic cache in Redis | `[cache]` settings |
| Input guardrail and security persona | automatic, `/persona load security` |
| Logging of every model call and monitoring dashboard | `[logging]` settings, `/dashboard [24h\|7d\|30d]` |
| Export and import of conversations as JSON | `/export current\|<uuid>`, `/import <path>` |

## Requirements

- [Ollama](https://ollama.com), which serves the models (see below)
- [uv](https://docs.astral.sh/uv/), which installs Python 3.14 and the dependencies
- `make`
- Docker, for the vector database the RAG indexes into and the Redis server of the semantic cache

## Installing Ollama

1. Install Ollama:

   ```sh
   curl -fsSL https://ollama.com/install.sh | sh     # Linux
   ```

   On macOS and Windows, download the installer from [ollama.com/download](https://ollama.com/download).

2. Start the server, if the installer did not start it as a service:

   ```sh
   ollama serve      # answers on http://localhost:11434
   ```

3. Pull a chat model and the embedding model:

   ```sh
   ollama pull qwen3:1.7b              # chat model used by `make run`
   ollama pull nomic-embed-text        # embedding model of the RAG, the web tools and the cache
   ```

   The embedding model is required: Sensai stops at start if Ollama does not serve the one set in
   `[llm].embedding_model`.

4. Check that both are there:

   ```sh
   ollama list
   ```

### Which models

The choice comes from our benchmarks (summary in [technical-notes.md](docs/architecture/technical-notes.md),
full tables in [benchmarks/models/results](benchmarks/models/results/README.md) and
[benchmarks/embeddings/results](benchmarks/embeddings/results/README.md)):

| Use | Model | Why |
|---|---|---|
| Chat, light default | `qwen3:1.7b` | 1.4 GB, fastest (161 tok/s), runs without a GPU, but picks the right tool in 5 cases of 11 |
| Chat, best results | `gemma4:e4b` | Only model passing every case (11/11 tools, 6/6 instructions); 6.6 GB, slow first load |
| Chat, runner-up | `qwen2.5:7b` | Every instruction followed, 8/11 tool cases |
| Embeddings, default | `nomic-embed-text` | 0.27 GB, small enough to stay loaded beside the chat model |
| Embeddings, better retrieval | `embeddinggemma` | Right passage first in 17/20 cases against 10/20, at 0.62 GB |

Any model Ollama serves works: pass it with `--model`, or switch with `/model` while Sensai runs. Models without the
`tools` capability answer, but cannot use any tool.

## Quick start

```sh
make                          # create .venv, install dependencies and the pre-commit hooks, copy settings.toml
make run                      # start Qdrant and Redis with Docker, then Sensai on qwen3:1.7b
make run MODEL=gemma4:e4b     # same, on another model
```

Without `make`:

```sh
docker compose up -d --wait
uv run sensai --model qwen3:1.7b
```

`uv sync --locked` installs the dependencies; a `requirements.txt` is also provided for `pip install -r
requirements.txt`.

## Configuration

Everything is set in `src/sensai/config/settings.toml`, which `make install` copies from `settings.toml.template`
(git-ignored, edit your copy). Without it, Sensai runs with the template; `--config <path>` loads another file.

| Table | What it sets |
|---|---|
| `[llm]`, `[llm.ollama]` | Embedding model, URL of the Ollama API |
| `[db]` | SQLite database of the conversations, location of Qdrant |
| `[context]` | Token budget, warning threshold, messages kept out of the compression |
| `[cache]` | Redis URL, similarity threshold and lifetime of the cached answers |
| `[web]` | Ollama web API key (optional, DuckDuckGo is used without it), timeout |
| `[features]` | Tools to disable, tools asking confirmation before they run |
| `[personas]` | Persona files loaded at start |
| `[logging]` | Record the model calls, with or without their content |
| `[mcp.<name>]` | MCP servers, see [MCP server configuration](#mcp-server-configuration) |

## Usage

Type a message and press Enter: the answer streams in. Commands start with `/`, and Tab completes them and their
arguments. `/help` lists them all.

```text
> What does EBITDA stand for?                          plain question, answered with the conversation as context
> /effort off                                          answer directly, without thinking steps or tools
> /effort high                                         think up to 5 steps, using the tools
> How did NVIDIA stock move this week?                 -> Using web_search... then an answer from the results

> /index reports/q3_results.pdf reports/forecast.md    index documents (text, Markdown, HTML, PDF, URLs)
> How does Q3 revenue compare with the forecast?       answer from the documents, ending with "Sources: ..."
> /ask What risks does the forecast list?              answer from the documents only

> /persona list                                        list the personas
> /persona load analyst                                switch persona, keeping the conversation
> /model                                               list the models Ollama serves
> /model qwen2.5:7b                                    switch model

> /branch list                                         list the stored conversations
> /branch set <uuid>                                   resume one
> /branch fork                                         copy the current one into a new branch
> /compress                                            summarise the oldest messages to free context
> /export current                                      write the conversation to <uuid>.json
> /import <path>                                       load an exported conversation

> /tools list                                          list the tools, and which ask confirmation
> /dashboard 7d                                        metrics of the model calls over 7 days
```

The input box shows the context used against the budget (top left) and the thinking effort (bottom right). Escape
or Ctrl+C stops an answer; Ctrl+Q quits.

Complete walkthroughs, from one use case to another, are the scenarios of
[docs/user_stories/analyst.md](docs/user_stories/analyst.md#4-user-scenarios).

## Vector database

`/index` and `/ask` store and read their chunks in Qdrant, which `make run` starts for you. On its own:

```sh
make docker         # start Qdrant and Redis
make docker-stop    # stop them
```

The documents live in the `qdrant_data` Docker volume, so a container stopped and started again finds them.
`docker compose down -v` is what drops them.

Without Docker, pointing `db.rag.location` in `settings.toml` at a directory (`"rag_qdrant"`) runs Qdrant embedded in
the process instead. It needs no server, but it locks that directory for one process at a time: the test suite and a
running sensai cannot share one.

## Semantic cache

Answers are cached in Redis, which `make run` starts along with Qdrant. The cached answers live in the `redis-data`
Docker volume. Without a reachable Redis server, sensai runs without the cache.

## MCP server configuration

Sensai starts every MCP server configured in the settings file, each in its own `[mcp.<name>]` table. The name
must only contain letters, digits, `_` or `-`. A server is either a local subprocess (`transport = "stdio"`, the
default) or a remote server reached over streamable HTTP (`transport = "http"`).

| Key | Transport | Default | Meaning |
|---|---|---|---|
| `transport` | both | `"stdio"` | `"stdio"` or `"http"` |
| `command`, `args` | stdio | required, `[]` | Program starting the server, and its arguments |
| `url` | http | required | `http(s)://` endpoint of the server, often ending in `/mcp` |
| `headers` | http | `{}` | Headers sent with every request, e.g. `{ Authorization = "Bearer ..." }` |
| `enabled` | both | `true` | `false` keeps the table without starting the server |
| `startup_timeout` | both | `10` | Seconds to wait for the server to start and list its tools |
| `call_timeout` | both | `30` | Seconds to wait for one tool call |

The tools of a server are named `<server>__<tool>`, e.g. `filesystem__list_allowed_directories`, so that two
servers can offer a tool of the same name. Use that name in `[features]` to disable a tool
(`disabled = ["tools.filesystem__write_file"]`) or to ask before it runs (`confirm = [...]`).

Servers start concurrently. A server that fails to start, or does not answer within its `startup_timeout`, is
skipped: Sensai starts without its tools and says why, e.g.
`MCP server 'finance' failed to start, its tools are unavailable: All connection attempts failed`.

### Your own server

The local deterministic test server exposes an `echo` tool:

```toml
[mcp.echo]
command = "uv"
args = ["run", "python", "tests/server/tools/mcp/fixtures/tiny_server.py"]
```

### A pre-existing server

The tested pre-existing MCP server is the official filesystem server. It exposes tools such as
`list_allowed_directories` and is started with `npx`:

```toml
[mcp.filesystem]
command = "npx"
args = ["--yes", "@modelcontextprotocol/server-filesystem", "/tmp/sensai-mcp-demo"]
startup_timeout = 30
```

The configured directory must exist and is the only directory the filesystem server may access.

### A remote server

The test server can also serve over HTTP, to try a remote server locally:

```bash
uv run python tests/server/tools/mcp/fixtures/tiny_server.py --http 8000   # serves http://127.0.0.1:8000/mcp
```

```toml
[mcp.remote]
transport = "http"
url = "http://127.0.0.1:8000/mcp"
headers = { Authorization = "Bearer <token>" }   # only if the server asks for one
```

Keep tokens in your git-ignored `settings.toml`, never in the template.

All the tables above can be used at once: Sensai then offers `echo__echo`, the `filesystem__` tools and
`remote__echo` side by side.

Each server keeps one session open from the start of Sensai until it quits: tool calls reuse it, so a server
keeps its state between calls (an open browser, a login) and a remote server is connected to only once. Current
limit: a server that disconnects while Sensai runs is not reconnected. Its tools then answer
`MCP server '<name>' disconnected, restart Sensai to reconnect: ...`, while the other servers keep working.

## Monitoring

The server records every call it makes to a model (each step of a ReAct exchange counts as one call) in the
`interaction` table of the persistent database: the model, the persona, the latency, the time to first token, the
token counts, the tools called and how the call ended (`ok`, `error` or `interrupted`). Answers served from the
semantic cache never reach the model and are not recorded. The `[logging]` table of `settings.toml` controls it:

| Key | Default | Description |
|---|---|---|
| `enabled` | `true` | `false` records nothing |
| `log_content` | `true` | `false` keeps the metrics and the tool names, and drops the prompt, the answer, the thinking and the tool arguments |

The records are written in the background and never delay an answer: a failed write is only logged on the server.
Stopping the server cleanly stops the running answers and waits for their records; a killed process loses the
records not yet written.

## Development

```sh
make tests    # run the test suite
make lint     # run every pre-commit check (same as the CI)
make format   # format the code and fix what ruff can fix
```

The coding style (120-character lines, every ruff rule enabled) is configured
in `pyproject.toml` under `[tool.ruff]`. The pre-commit hooks, defined in
`.pre-commit-config.yaml`, run on every commit:

- ruff check and ruff format
- trailing whitespace, end of file, YAML/TOML syntax, merge conflicts, large
  files
- [Conventional Commits](https://www.conventionalcommits.org/) format for
  commit messages (`feat: ...`, `fix: ...`)

The CI rejects any code that does not pass these checks or the tests.

## Make targets

| Target        | Description                                                         |
|---------------|---------------------------------------------------------------------|
| `all`         | Default. Runs `install`                                             |
| `install`     | `uv sync --locked`, the pre-commit hooks, and a copy of the settings |
| `run`         | Starts Qdrant and Redis, then Sensai on `MODEL` (`qwen3:1.7b`)      |
| `docker`      | Starts Qdrant and Redis with Docker Compose                         |
| `docker-stop` | Stops them                                                          |
| `help`        | Shows the command line options                                      |
| `tests`       | Runs the test suite with `pytest`                                   |
| `lint`        | Runs every pre-commit check on all files                            |
| `format`      | Runs `ruff format` and `ruff check --fix`                           |
| `lock`        | Regenerates `uv.lock` from `pyproject.toml`                         |

## Dependencies

Dependencies are declared in `pyproject.toml` and locked in `uv.lock`:

- `[project].dependencies`: runtime dependencies, needed to run the
  program. These are the only ones included when the package is built.
- `[dependency-groups].dev`: development tools (pytest, ruff, pre-commit).
- `uv.lock`: every package, including transitive ones, at an exact version.
  It is generated, do not edit it, but commit it.

`uv sync` installs both runtime and dev dependencies. Use
`uv sync --no-dev` to install only what the program needs.

To add or upgrade a dependency:

```sh
uv add <package>                  # runtime dependency
uv add --dev <package>            # dev dependency
uv lock --upgrade-package <pkg>   # upgrade one package
```

Then commit `pyproject.toml` and `uv.lock`. `make install` (and the CI) use
`--locked`: they fail if `uv.lock` is out of date instead of resolving new
versions.

## Project layout

```
.
├── pyproject.toml            # project metadata, dependencies, tool config
├── uv.lock                   # locked dependencies (generated)
├── requirements.txt          # the same dependencies, for pip
├── Makefile
├── compose.yaml              # Qdrant and Redis
├── .github/workflows/        # CI: install, run, style, tests + mirroring
├── src/                      # the code of Sensai
├── tests/                    # pytest test suite, Ollama mocked
├── benchmarks/               # results of the chat and embedding model benchmarks
└── docs/
    ├── features.md           # implemented features, and their traceability to stories, code and tests
    ├── user_stories/         # user stories and user scenarios
    └── architecture/         # architecture, technical choices, database schema
```

## Contributors

<table>
  <tr>
    <td align="center">
      <a href="https://github.com/amosAlmacin">
        <img src="https://github.com/amosAlmacin.png?size=100" width="100px;" alt="Amos Almacin"/>
        <br />
        <sub><b>Amos Almacin</b></sub>
      </a>
    </td>
    <td align="center">
      <a href="https://github.com/Dimitri-Collineau">
        <img src="https://github.com/Dimitri-Collineau.png?size=100" width="100px;" alt="Dimitri-Collineau"/>
        <br />
        <sub><b>Dimitri Collineau</b></sub>
      </a>
    </td>
    <td align="center">
      <a href="https://github.com/EstebanPeroz">
        <img src="https://github.com/EstebanPeroz.png?size=100" width="100px;" alt="EstebanPeroz"/>
        <br />
        <sub><b>Esteban Peroz</b></sub>
      </a>
    </td>
    <td align="center">
      <a href="https://github.com/EwanCz">
        <img src="https://github.com/EwanCz.png?size=100" width="100px;" alt="EwanCz"/>
        <br />
        <sub><b>Ewan Czarny</b></sub>
      </a>
    </td>
    <td align="center">
      <a href="https://github.com/N0AH-sve">
        <img src="https://github.com/N0AH-sve.png?size=100" width="100px;" alt="N0AH-sve"/>
        <br />
        <sub><b>Noah Savoye</b></sub>
      </a>
    </td>
  </tr>
</table>
