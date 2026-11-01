# Sensai server

The Sensai logic, served to the clients over a websocket. Start it with `make run-server`, which reads
`packages/server/settings.toml` (copied from `settings.toml.template`).

## Context budget

The `[context]` table sets how much of the model's context a conversation may take. The client shows the usage of
the current branch as a gauge; past the warning threshold, the next prompt first asks whether to compress the
conversation or to start a new one, and `/compress` summarises it by hand at any time.

| Key | Default | Meaning |
|---|---|---|
| `budget` | none | Token budget of a conversation, also sent to the model as its context window (`num_ctx`). Without it, the budget is the model's context length, and Ollama keeps its own, often much smaller, window. |
| `warning_threshold` | `0.8` | Share of the budget, in `]0, 1]`, past which the usage is flagged and the oldest messages are summarised. |
| `keep_recent` | `6` | Number of the last messages always sent as they are, never summarised. |

```toml
[context]
budget = 8192
warning_threshold = 0.8
keep_recent = 6
```

The usage is the token count of the branch's last model call, kept in memory: after a restart, a branch's usage is
unknown until its next call. After a compression, it is estimated (the summarised tokens taken off, the summary's
added) until the next call measures it again. A branch keeps one summary, stored in the `summary` table and not exported. The summary
call is logged like any other, with the persona but without the branch.

## MCP servers

The server offers the tools of every MCP server configured in its settings, one `[mcp.<name>]` table each. The name
must only contain letters, digits, `_` or `-`. A server is either a local subprocess (`transport = "stdio"`, the
default) or a remote server reached over streamable HTTP (`transport = "http"`).

| Key | Transport | Default | Meaning |
|---|---|---|---|
| `transport` | both | `"stdio"` | `"stdio"` or `"http"` |
| `command` | stdio | required | Program starting the server |
| `args` | stdio | `[]` | Its arguments |
| `url` | http | required | `http(s)://` endpoint of the server, often ending in `/mcp` |
| `headers` | http | `{}` | Headers sent with every request, e.g. `{ Authorization = "Bearer ..." }` |
| `enabled` | both | `true` | `false` keeps the table without starting the server |
| `startup_timeout` | both | `10` | Seconds to wait for the server to start and list its tools |
| `call_timeout` | both | `30` | Seconds to wait for one tool call |

```toml
[mcp.filesystem]
command = "npx"
args = ["--yes", "@modelcontextprotocol/server-filesystem", "/tmp/sensai-mcp-demo"]
startup_timeout = 30

[mcp.remote]
transport = "http"
url = "http://127.0.0.1:8000/mcp"
headers = { Authorization = "Bearer <token>" }   # keep tokens in settings.toml, never in the template
```

The tools of a server are named `<server>__<tool>`, e.g. `filesystem__list_allowed_directories`. Use that name in
`[features]` to disable a tool (`disabled = ["tools.filesystem__write_file"]`) or to ask the user before it runs
(`confirm = [...]`). An MCP tool runs without asking unless it is listed there.

The servers start concurrently with the app, and each keeps one session open until the app stops. A server that fails
to start is logged and left out, without stopping the app. A server that disconnects is not reconnected: its tools
answer `error: MCP server '<name>' disconnected…` until the app restarts. Only text results are supported.

`tests/mcp/fixtures/tiny_server.py` is a small test server, served over stdio, or over HTTP at
`http://127.0.0.1:PORT/mcp` with `--http PORT`.
