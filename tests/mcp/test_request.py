from sensai.mcp.request import add_tool_schemas


def test_add_tool_schemas_adds_enabled_schemas() -> None:
    payload: dict = {"model": "test", "messages": []}
    tool_manager = type("ToolManager", (), {"get_tool_call_rules": lambda _: [{"type": "function"}]})()

    add_tool_schemas(payload, tool_manager)

    assert payload == {
        "model": "test",
        "messages": [],
        "tools": [{"type": "function"}],
    }


def test_add_tool_schemas_leaves_payload_unchanged_without_tools() -> None:
    payload: dict = {"model": "test", "messages": []}
    tool_manager = type("ToolManager", (), {"get_tool_call_rules": lambda _: []})()

    add_tool_schemas(payload, tool_manager)

    assert payload == {"model": "test", "messages": []}
