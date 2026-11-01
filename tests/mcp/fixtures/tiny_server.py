import asyncio
import sys

from mcp.server.mcpserver import MCPServer

server = MCPServer("tiny-test-server")


@server.tool(description="Echo a message")
def echo(message: str) -> str:
    return message


@server.tool(description="Show a message in the terminal where this server runs, instead of sending it back")
def reverse_echo(message: str) -> str:
    # stderr, as stdout carries the MCP messages when the server runs over stdio.
    print(f"[reverse_echo] {message}", file=sys.stderr, flush=True)
    return "The message was shown in the terminal of the server."


if __name__ == "__main__":
    # `tiny_server.py --http PORT` serves over streamable HTTP at http://127.0.0.1:PORT/mcp instead of stdio.
    if len(sys.argv) == 3 and sys.argv[1] == "--http":
        asyncio.run(server.run_streamable_http_async(port=int(sys.argv[2])))
    else:
        asyncio.run(server.run_stdio_async())
