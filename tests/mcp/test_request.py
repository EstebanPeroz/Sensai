from sensai.mcp.request import add_tool_schemas, execute_one_tool_call


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


def test_execute_one_tool_call_returns_the_first_tool_result() -> None:
    result = {"role": "tool", "tool_name": "echo", "content": "hello"}
    tool_manager = type("ToolManager", (), {"call": lambda *_: result})()

    assert execute_one_tool_call([{"function": {"name": "echo"}}], tool_manager) == result


def test_execute_one_tool_call_does_not_call_manager_without_calls() -> None:
    class UnusedToolManager:
        def call(self, tool_call: dict) -> dict:
            del tool_call
            raise AssertionError

    assert execute_one_tool_call([], UnusedToolManager()) is None
