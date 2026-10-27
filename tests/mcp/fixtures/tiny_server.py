import asyncio

from mcp.server.mcpserver import MCPServer

server = MCPServer("tiny-test-server")


@server.tool(description="Echo a message")
def echo(message: str) -> str:
    return message


if __name__ == "__main__":
    asyncio.run(server.run_stdio_async())
