from sensai.ui.completion import Completer


def _completer(calls: list[list[str]] | None = None) -> Completer:
    def model(args: list[str]) -> list[str]:
        if calls is not None:
            calls.append(args)
        return ["llama3:8b", "qwen2.5:1.5b", "qwen3:1.7b"]

    return Completer({"/help": lambda _: [], "/model": model})


class TestComplete:
    def test_slash_alone_lists_every_command(self) -> None:
        assert _completer().complete("/") == ["/help", "/model"]

    def test_partial_command(self) -> None:
        assert _completer().complete("/mo") == ["/model"]

    def test_full_command_completes_to_itself(self) -> None:
        assert _completer().complete("/help") == ["/help"]

    def test_unknown_command_prefix(self) -> None:
        assert _completer().complete("/x") == []

    def test_command_and_space_lists_every_argument(self) -> None:
        assert _completer().complete("/model ") == [
            "/model llama3:8b",
            "/model qwen2.5:1.5b",
            "/model qwen3:1.7b",
        ]

    def test_partial_argument(self) -> None:
        assert _completer().complete("/model qwen3") == ["/model qwen3:1.7b"]

    def test_argument_of_command_without_values(self) -> None:
        assert _completer().complete("/help a") == []

    def test_argument_of_unknown_command(self) -> None:
        assert _completer().complete("/nope a") == []

    def test_plain_text_is_not_completed(self) -> None:
        assert _completer().complete("hello") == []

    def test_empty_text_is_not_completed(self) -> None:
        assert _completer().complete("") == []


class TestFetching:
    def test_command_name_does_not_fetch(self) -> None:
        calls: list[list[str]] = []
        _completer(calls).complete("/mod")
        assert calls == []

    def test_fetched_once_while_typing_a_word(self) -> None:
        calls: list[list[str]] = []
        completer = _completer(calls)
        for text in ("/model ", "/model q", "/model qw", "/model qwen"):
            completer.complete(text)
        assert calls == [[]]

    def test_fetched_again_when_a_word_starts(self) -> None:
        calls: list[list[str]] = []
        completer = _completer(calls)
        completer.complete("/model qwen3:1.7b")
        completer.complete("/model qwen3:1.7b ")
        assert calls == [[], ["qwen3:1.7b"]]

    def test_fetched_again_when_a_word_is_removed(self) -> None:
        calls: list[list[str]] = []
        completer = _completer(calls)
        completer.complete("/model a ")
        completer.complete("/model a")
        assert calls == [["a"], []]

    def test_set_sources_forgets_fetched_values(self) -> None:
        calls: list[list[str]] = []
        completer = _completer(calls)
        completer.complete("/model ")
        completer.set_sources({"/model": lambda _: ["new"]})
        assert completer.complete("/model ") == ["/model new"]
