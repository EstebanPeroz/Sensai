# Sensai server

The Sensai logic, served to the clients over a websocket. Start it with `make run-server`, which reads
`packages/server/settings.toml` (copied from `settings.toml.template`).

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
