# Sensai

## Requirements

- [uv](https://docs.astral.sh/uv/)
- `make`
- Docker, for the vector database the RAG indexes into and the Redis server of the semantic cache

## Quick start

```sh
make          # create .venv, install dependencies and the pre-commit hooks
make run      # run the program
```

## Vector database

`/index` and `/ask` store and read their chunks in Qdrant, which `make run`
starts for you. On its own:

```sh
make qdrant
make qdrant-stop
```

The documents live in the `sensai_qdrant_data` Docker volume, so a container
stopped and started again finds them. `docker compose down -v` is what drops
them.

Without Docker, pointing `db.rag.location` in `settings.toml` at a directory
(`"rag_qdrant"`) runs Qdrant embedded in the process instead. It needs no
server, but it locks that directory for one process at a time: the test suite
and a running sensai cannot share one.

## Semantic cache

Answers are cached in Redis, which `make run` starts along with Qdrant (`make qdrant` and `make qdrant-stop` start and
stop both). The cached answers live in the `sensai_redis-data` Docker volume. Without a reachable Redis server, sensai
runs without the cache.

You can also call uv directly, without activating the virtualenv:

```sh
uv run sensai             # run the program
```

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
args = ["run", "python", "tests/mcp/fixtures/tiny_server.py"]
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
uv run python tests/mcp/fixtures/tiny_server.py --http 8000   # serves http://127.0.0.1:8000/mcp
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

| Target    | Description                                                     |
|-----------|-----------------------------------------------------------------|
| `all`     | Default. Runs `install`                                         |
| `install` | `uv sync --locked`, then installs the pre-commit hooks          |
| `run`     | Runs the program                                                |
| `tests`   | Runs the test suite with `pytest`                               |
| `lint`    | Runs every pre-commit check on all files                        |
| `format`  | Runs `ruff format` and `ruff check --fix`                       |
| `lock`    | Regenerates `uv.lock` from `pyproject.toml`                     |

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
├── .python-version           # Python version used by uv
├── .pre-commit-config.yaml   # pre-commit hooks
├── Makefile
├── .github/workflows/        # CI: install, run, style, tests + mirroring
├── src/
│   └── sensai/               # the package
│       ├── __init__.py
│       ├── __main__.py       # `python -m sensai`, delegates to main()
│       ├── main.py           # entry point and composition root: main()
│       └── parsing.py        # command line arguments
├── tests/                    # pytest test suite
└── docs/
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
