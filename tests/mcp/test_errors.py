from sensai.mcp.errors import MCPStartupError, describe


class TestMCPStartupErrorFromError:
    def test_describes_a_timeout_with_the_waited_time(self) -> None:
        assert str(MCPStartupError.from_error(TimeoutError(), 2.5)) == "no answer within 2.5s"

    def test_keeps_the_message_of_another_error(self) -> None:
        assert str(MCPStartupError.from_error(OSError("connection refused"), 10)) == "connection refused"

    def test_falls_back_to_the_error_type_without_a_message(self) -> None:
        assert str(MCPStartupError.from_error(EOFError(), 10)) == "EOFError"

    def test_describes_an_exception_group_by_its_first_failure(self) -> None:
        group = BaseExceptionGroup("unhandled errors", [KeyboardInterrupt(), OSError("connection refused")])

        assert str(MCPStartupError.from_error(BaseExceptionGroup("outer", [group]), 10)) == "connection refused"


class TestDescribe:
    def test_uses_the_message(self) -> None:
        assert describe(OSError("connection refused")) == "connection refused"

    def test_falls_back_to_the_type_without_a_message(self) -> None:
        assert describe(EOFError()) == "EOFError"

    def test_describes_a_nested_group_by_its_first_failure(self) -> None:
        inner = BaseExceptionGroup("inner", [KeyboardInterrupt(), OSError("connection refused")])

        assert describe(ExceptionGroup("outer", [ValueError("first")])) == "first"
        assert describe(BaseExceptionGroup("outer", [inner])) == "connection refused"
