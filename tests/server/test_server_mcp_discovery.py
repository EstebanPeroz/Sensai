import dataclasses

import pytest
from sensai_server.tools.mcp.discovery import MCPToolInfo


def test_keeps_name_description_and_input_schema() -> None:
    schema = {"type": "object", "properties": {"message": {"type": "string"}}}

    info = MCPToolInfo(name="echo", description="Echo a message", input_schema=schema)

    assert (info.name, info.description, info.input_schema) == ("echo", "Echo a message", schema)


def test_is_frozen() -> None:
    info = MCPToolInfo(name="echo", description=None, input_schema={})

    with pytest.raises(dataclasses.FrozenInstanceError):
        info.name = "other"  # type: ignore[misc]
