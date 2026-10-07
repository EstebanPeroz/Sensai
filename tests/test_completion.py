from sensai.ui.completion import complete

COMPLETIONS = {
    "/help": [],
    "/model": ["llama3:8b", "qwen2.5:1.5b", "qwen3:1.7b"],
}


class TestComplete:
    def test_slash_alone_lists_every_command(self) -> None:
        assert complete("/", COMPLETIONS) == ["/help", "/model"]

    def test_partial_command(self) -> None:
        assert complete("/mo", COMPLETIONS) == ["/model"]

    def test_full_command_completes_to_itself(self) -> None:
        assert complete("/help", COMPLETIONS) == ["/help"]

    def test_unknown_command_prefix(self) -> None:
        assert complete("/x", COMPLETIONS) == []

    def test_command_and_space_lists_every_argument(self) -> None:
        assert complete("/model ", COMPLETIONS) == ["/model llama3:8b", "/model qwen2.5:1.5b", "/model qwen3:1.7b"]

    def test_partial_argument(self) -> None:
        assert complete("/model qwen3", COMPLETIONS) == ["/model qwen3:1.7b"]

    def test_argument_of_command_without_values(self) -> None:
        assert complete("/help a", COMPLETIONS) == []

    def test_argument_of_unknown_command(self) -> None:
        assert complete("/nope a", COMPLETIONS) == []

    def test_plain_text_is_not_completed(self) -> None:
        assert complete("hello", COMPLETIONS) == []

    def test_empty_text_is_not_completed(self) -> None:
        assert complete("", COMPLETIONS) == []
