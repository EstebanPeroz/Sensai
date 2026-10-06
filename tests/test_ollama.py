import json
from collections.abc import Iterator
from unittest.mock import MagicMock, patch

import pytest
import requests

from sensai.config.settings import OllamaSettings
from sensai.llm.error import JsonError, RequestCallError, RequestStatusError
from sensai.llm.ollama import OllamaAdapter
from sensai.llm.responses import ChatResponse, ShowResponse

_API_PATH = "http://localhost:11434/"
_EMBEDDING_MODEL = "nomic-embed-text"


@pytest.fixture
def adapter() -> OllamaAdapter:
    return OllamaAdapter(OllamaSettings())


def mock_response(
    *,
    ok: bool = True,
    status_code: int = 200,
    json_data: dict | None = None,
    json_error: bool = False,
    lines: list[str] | None = None,
) -> MagicMock:
    response = MagicMock(spec=requests.Response)
    response.ok = ok
    response.status_code = status_code
    if json_error:
        response.json.side_effect = requests.exceptions.JSONDecodeError("Expecting value", "", 0)
    else:
        response.json.return_value = json_data
    response.iter_lines.return_value = iter(lines or [])
    return response


class TestSettings:
    @pytest.mark.parametrize("base_url", ["http://ollama:1234", "http://ollama:1234/"])
    def test_calls_the_configured_base_url(self, base_url: str) -> None:
        adapter = OllamaAdapter(OllamaSettings(base_url=base_url))
        with patch("sensai.llm.ollama.requests.post", return_value=mock_response(json_data={})) as mock_post:
            adapter.show("model")

        assert mock_post.call_args.args == ("http://ollama:1234/api/show",)


class TestChatNonStreaming:
    def test_returns_chat_response_on_success(self, adapter: OllamaAdapter) -> None:
        payload = {
            "model": "qwen2.5:1.5b",
            "message": {"role": "assistant", "content": "hello", "thinking": "", "tool_calls": []},
            "done": True,
        }
        with patch("sensai.llm.ollama.requests.post", return_value=mock_response(json_data=payload)) as mock_post:
            result = adapter.chat({"model": "qwen2.5:1.5b", "messages": []}, stream=False)

        assert isinstance(result, ChatResponse)
        assert result.content == "hello"
        assert result.role == "assistant"
        assert result.done is True
        mock_post.assert_called_once()
        assert mock_post.call_args.args == (_API_PATH + "api/chat",)
        assert mock_post.call_args.kwargs["json"]["stream"] is False

    def test_raises_request_call_error_on_request_exception(self, adapter: OllamaAdapter) -> None:
        with (
            patch("sensai.llm.ollama.requests.post", side_effect=requests.RequestException),
            pytest.raises(RequestCallError),
        ):
            adapter.chat({"model": "x", "messages": []}, stream=False)

    def test_raises_request_status_error_on_non_ok_status(self, adapter: OllamaAdapter) -> None:
        with (
            patch("sensai.llm.ollama.requests.post", return_value=mock_response(ok=False, status_code=500)),
            pytest.raises(RequestStatusError),
        ):
            adapter.chat({"model": "x", "messages": []}, stream=False)

    def test_raises_json_error_on_invalid_json(self, adapter: OllamaAdapter) -> None:
        with (
            patch("sensai.llm.ollama.requests.post", return_value=mock_response(json_error=True)),
            pytest.raises(JsonError),
        ):
            adapter.chat({"model": "x", "messages": []}, stream=False)


class TestChatStreaming:
    def test_yields_one_chat_response_per_line(self, adapter: OllamaAdapter) -> None:
        lines = [
            json.dumps({"model": "m", "message": {"role": "assistant", "content": "The "}, "done": False}),
            "",
            "not-json",
            json.dumps({"model": "m", "message": {"role": "assistant", "content": "sky"}, "done": False}),
            json.dumps({"model": "m", "message": {"role": "assistant", "content": ""}, "done": True}),
        ]
        with patch("sensai.llm.ollama.requests.post", return_value=mock_response(lines=lines)) as mock_post:
            result = adapter.chat({"model": "m", "messages": []}, stream=True)
            chunks = list(result)

        assert [chunk.content for chunk in chunks] == ["The ", "sky", None]
        assert [chunk.done for chunk in chunks] == [False, False, True]
        assert mock_post.call_args.kwargs["stream"] is True
        assert mock_post.call_args.kwargs["json"]["stream"] is True

    def test_raises_request_call_error_on_request_exception(self, adapter: OllamaAdapter) -> None:
        with (
            patch("sensai.llm.ollama.requests.post", side_effect=requests.RequestException),
            pytest.raises(RequestCallError),
        ):
            adapter.chat({"model": "x", "messages": []}, stream=True)

    def test_raises_request_status_error_on_non_ok_status(self, adapter: OllamaAdapter) -> None:
        with (
            patch("sensai.llm.ollama.requests.post", return_value=mock_response(ok=False, status_code=404)),
            pytest.raises(RequestStatusError),
        ):
            adapter.chat({"model": "x", "messages": []}, stream=True)

    def test_closes_response_after_full_consumption(self, adapter: OllamaAdapter) -> None:
        lines = [json.dumps({"model": "m", "message": {"role": "assistant", "content": "hi"}, "done": True})]
        response = mock_response(lines=lines)
        with patch("sensai.llm.ollama.requests.post", return_value=response):
            result = adapter.chat({"model": "m", "messages": []}, stream=True)
            list(result)

        response.close.assert_called_once()

    def test_closes_response_on_early_termination(self, adapter: OllamaAdapter) -> None:
        lines = [
            json.dumps({"model": "m", "message": {"role": "assistant", "content": "a"}, "done": False}),
            json.dumps({"model": "m", "message": {"role": "assistant", "content": "b"}, "done": False}),
        ]
        response = mock_response(lines=lines)
        with patch("sensai.llm.ollama.requests.post", return_value=response):
            result = adapter.chat({"model": "m", "messages": []}, stream=True)
            next(result)
            result.close()

        response.close.assert_called_once()

    def test_raises_request_call_error_on_read_error_mid_stream(self, adapter: OllamaAdapter) -> None:
        def raising_lines(*_args: object, **_kwargs: object) -> Iterator[str]:
            yield json.dumps({"model": "m", "message": {"role": "assistant", "content": "a"}, "done": False})
            raise requests.exceptions.ChunkedEncodingError

        def drain(it: Iterator[ChatResponse], sink: list[ChatResponse]) -> None:
            for chunk in it:
                sink.append(chunk)  # noqa: PERF402 -- partial results must survive the raised error

        response = mock_response(lines=[])
        response.iter_lines.side_effect = raising_lines
        with patch("sensai.llm.ollama.requests.post", return_value=response):
            result = adapter.chat({"model": "m", "messages": []}, stream=True)
            chunks: list[ChatResponse] = []
            with pytest.raises(RequestCallError):
                drain(result, chunks)

        assert [chunk.content for chunk in chunks] == ["a"]
        response.close.assert_called_once()


class TestEmbedding:
    def test_embedding_returns_first_vector(self, adapter: OllamaAdapter) -> None:
        with patch(
            "sensai.llm.ollama.requests.post",
            return_value=mock_response(json_data={"embeddings": [[0.1, 0.2]]}),
        ):
            assert adapter.embedding("hello", _EMBEDDING_MODEL) == [0.1, 0.2]

    def test_embedding_returns_empty_list_when_no_embeddings(self, adapter: OllamaAdapter) -> None:
        with patch("sensai.llm.ollama.requests.post", return_value=mock_response(json_data={"embeddings": []})):
            assert adapter.embedding("hello", _EMBEDDING_MODEL) == []

    def test_embeddings_returns_vectors(self, adapter: OllamaAdapter) -> None:
        with patch(
            "sensai.llm.ollama.requests.post",
            return_value=mock_response(json_data={"embeddings": [[0.1], [0.2]]}),
        ) as mock_post:
            assert adapter.embeddings(["a", "b"], _EMBEDDING_MODEL) == [[0.1], [0.2]]
        assert mock_post.call_args.args == (_API_PATH + "api/embed",)
        assert mock_post.call_args.kwargs["json"] == {"model": _EMBEDDING_MODEL, "input": ["a", "b"]}

    def test_embeddings_returns_empty_list_when_model_unavailable(self, adapter: OllamaAdapter) -> None:
        """embeddings() probes the model via show() first, and bails out without embedding if that fails."""
        with patch(
            "sensai.llm.ollama.requests.post",
            return_value=mock_response(ok=False, status_code=404),
        ) as mock_post:
            assert adapter.embeddings(["a"], _EMBEDDING_MODEL) == []
        mock_post.assert_called_once()
        assert mock_post.call_args.args[0] == _API_PATH + "api/show"

    def test_embeddings_raises_request_call_error_on_call_failure(self, adapter: OllamaAdapter) -> None:
        def side_effect(url: str, **_kwargs: object) -> MagicMock:
            if url == _API_PATH + "api/show":
                return mock_response(json_data={})
            raise requests.RequestException

        with (
            patch("sensai.llm.ollama.requests.post", side_effect=side_effect),
            pytest.raises(RequestCallError),
        ):
            adapter.embeddings(["a", "b"], _EMBEDDING_MODEL)


class TestShow:
    def test_raises_request_status_error_when_model_not_found(self, adapter: OllamaAdapter) -> None:
        with (
            patch("sensai.llm.ollama.requests.post", return_value=mock_response(ok=False, status_code=404)),
            pytest.raises(RequestStatusError),
        ):
            adapter.show("missing-model")

    def test_returns_response_with_no_capabilities(self, adapter: OllamaAdapter) -> None:
        with patch("sensai.llm.ollama.requests.post", return_value=mock_response(json_data={})):
            result = adapter.show("model")

        assert isinstance(result, ShowResponse)
        assert result.model == "model"
        assert result.tools is False
        assert result.think is False

    def test_detects_tools_and_thinking_capabilities(self, adapter: OllamaAdapter) -> None:
        with patch(
            "sensai.llm.ollama.requests.post",
            return_value=mock_response(json_data={"capabilities": ["tools", "thinking"]}),
        ):
            result = adapter.show("model")

        assert result.tools is True
        assert result.think is True


class TestLoad:
    def test_returns_true_when_done(self, adapter: OllamaAdapter) -> None:
        with patch("sensai.llm.ollama.requests.post", return_value=mock_response(json_data={"done": True})):
            assert adapter.load("model") is True

    def test_returns_false_when_not_done(self, adapter: OllamaAdapter) -> None:
        with patch("sensai.llm.ollama.requests.post", return_value=mock_response(json_data={"done": False})):
            assert adapter.load("model") is False

    def test_raises_request_call_error_on_call_failure(self, adapter: OllamaAdapter) -> None:
        with (
            patch("sensai.llm.ollama.requests.post", side_effect=requests.RequestException),
            pytest.raises(RequestCallError),
        ):
            adapter.load("model")


class TestUnload:
    def test_returns_true_when_done(self, adapter: OllamaAdapter) -> None:
        with patch(
            "sensai.llm.ollama.requests.post",
            return_value=mock_response(json_data={"done": True}),
        ) as mock_post:
            assert adapter.unload("model") is True
        assert mock_post.call_args.kwargs["json"]["keep_alive"] == 0

    def test_raises_request_call_error_on_call_failure(self, adapter: OllamaAdapter) -> None:
        with (
            patch("sensai.llm.ollama.requests.post", side_effect=requests.RequestException),
            pytest.raises(RequestCallError),
        ):
            adapter.unload("model")


class TestList:
    def test_returns_model_names(self, adapter: OllamaAdapter) -> None:
        with patch(
            "sensai.llm.ollama.requests.get",
            return_value=mock_response(json_data={"models": [{"name": "a"}, {"name": "b"}]}),
        ):
            assert adapter.list() == ["a", "b"]

    def test_returns_empty_list_when_no_models(self, adapter: OllamaAdapter) -> None:
        with patch("sensai.llm.ollama.requests.get", return_value=mock_response(json_data={"models": []})):
            assert adapter.list() == []

    def test_raises_request_call_error_on_call_failure(self, adapter: OllamaAdapter) -> None:
        with (
            patch("sensai.llm.ollama.requests.get", side_effect=requests.RequestException),
            pytest.raises(RequestCallError),
        ):
            adapter.list()

    def test_raises_request_status_error_on_non_ok_status(self, adapter: OllamaAdapter) -> None:
        with (
            patch("sensai.llm.ollama.requests.get", return_value=mock_response(ok=False, status_code=500)),
            pytest.raises(RequestStatusError),
        ):
            adapter.list()
