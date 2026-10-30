import asyncio
import sys

from mcp.server.mcpserver import MCPServer

server = MCPServer("tiny-test-server")


@server.tool(description="Echo a message")
def echo(message: str) -> str:
    return message


if __name__ == "__main__":
    # `tiny_server.py --http PORT` serves over streamable HTTP at http://127.0.0.1:PORT/mcp instead of stdio.
    if len(sys.argv) == 3 and sys.argv[1] == "--http":
        asyncio.run(server.run_streamable_http_async(port=int(sys.argv[2])))
    else:
        asyncio.run(server.run_stdio_async())
