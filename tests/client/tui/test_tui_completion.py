from sensai_tui.completion import Completer

MODELS = ["llama3:8b", "qwen2.5:1.5b", "qwen3:1.7b"]


def _completer(requests: list[str] | None = None) -> Completer:
    """A completer whose server answers at once: the models for /model, nothing for the other commands."""

    def request(text: str) -> None:
        if requests is not None:
            requests.append(text)
        completer.receive(text, MODELS if text.startswith("/model") else [])

    completer = Completer(request, ["/help", "/model"])
    return completer


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

    def test_no_commands_before_session(self) -> None:
        assert Completer().complete("/") == []


class TestRequests:
    def test_command_name_does_not_request(self) -> None:
        requests: list[str] = []
        _completer(requests).complete("/mod")
        assert requests == []

    def test_unknown_command_does_not_request(self) -> None:
        requests: list[str] = []
        _completer(requests).complete("/nope a")
        assert requests == []

    def test_requested_once_while_typing_a_word(self) -> None:
        requests: list[str] = []
        completer = _completer(requests)
        for text in ("/model ", "/model q", "/model qw", "/model qwen"):
            completer.complete(text)
        assert requests == ["/model "]

    def test_requested_again_when_a_word_starts(self) -> None:
        requests: list[str] = []
        completer = _completer(requests)
        completer.complete("/model qwen3:1.7b")
        completer.complete("/model qwen3:1.7b ")
        assert requests == ["/model qwen3:1.7b", "/model qwen3:1.7b "]

    def test_requested_again_when_a_word_is_removed(self) -> None:
        requests: list[str] = []
        completer = _completer(requests)
        completer.complete("/model a ")
        completer.complete("/model a")
        assert requests == ["/model a ", "/model a"]

    def test_set_commands_forgets_received_values(self) -> None:
        requests: list[str] = []
        completer = _completer(requests)
        completer.complete("/model ")
        completer.set_commands(["/model"])
        completer.complete("/model ")
        assert requests == ["/model ", "/model "]


class TestReceive:
    def test_values_arrive_after_the_request(self) -> None:
        requests: list[str] = []
        completer = Completer(requests.append, ["/model"])
        assert completer.complete("/model ") == []
        assert completer.receive(requests[0], MODELS)
        assert completer.complete("/model qwen") == ["/model qwen2.5:1.5b", "/model qwen3:1.7b"]

    def test_partial_word_in_the_answer_keeps_the_key(self) -> None:
        completer = Completer(lambda _: None, ["/model"])
        completer.complete("/model ")
        assert completer.receive("/model q", MODELS)

    def test_late_answer_is_dropped(self) -> None:
        completer = Completer(lambda _: None, ["/persona"])
        completer.complete("/persona ")
        completer.complete("/persona load ")
        assert not completer.receive("/persona ", ["load", "list"])
        assert completer.complete("/persona load ") == []

    def test_answer_without_request_is_dropped(self) -> None:
        completer = Completer(lambda _: None, ["/model"])
        assert not completer.receive("/model ", MODELS)
        assert not completer.receive("/model", MODELS)
        assert not completer.receive("", MODELS)
