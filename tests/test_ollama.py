import json
from collections.abc import Iterator
from unittest.mock import MagicMock, patch

import pytest
import requests

from sensai.config.settings import OllamaSettings
from sensai.llm.ollama import OllamaAdapter
from sensai.llm.responses import ChatResponse, ShowResponse

_API_PATH = "http://localhost:11434/"


@pytest.fixture
def adapter() -> OllamaAdapter:
    return OllamaAdapter(OllamaSettings())


def mock_response(
    *,
    ok: bool = True,
    json_data: dict | None = None,
    json_error: bool = False,
    lines: list[str] | None = None,
) -> MagicMock:
    response = MagicMock(spec=requests.Response)
    response.ok = ok
    if json_error:
        response.json.side_effect = requests.exceptions.JSONDecodeError("Expecting value", "", 0)
    else:
        response.json.return_value = json_data
    response.iter_lines.return_value = iter(lines or [])
    return response


class TestSettings:
    def test_uses_configured_base_url(self) -> None:
        adapter = OllamaAdapter(OllamaSettings(base_url="http://ollama:1234"))
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

    def test_returns_none_on_request_exception(self, adapter: OllamaAdapter) -> None:
        with patch("sensai.llm.ollama.requests.post", side_effect=requests.RequestException):
            assert adapter.chat({"model": "x", "messages": []}, stream=False) is None

    def test_returns_none_on_non_ok_status(self, adapter: OllamaAdapter) -> None:
        with patch("sensai.llm.ollama.requests.post", return_value=mock_response(ok=False)):
            assert adapter.chat({"model": "x", "messages": []}, stream=False) is None

    def test_returns_none_on_invalid_json(self, adapter: OllamaAdapter) -> None:
        with patch("sensai.llm.ollama.requests.post", return_value=mock_response(json_error=True)):
            assert adapter.chat({"model": "x", "messages": []}, stream=False) is None


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
            assert result is not None
            chunks = list(result)

        assert [chunk.content for chunk in chunks] == ["The ", "sky", ""]
        assert [chunk.done for chunk in chunks] == [False, False, True]
        assert mock_post.call_args.kwargs["stream"] is True
        assert mock_post.call_args.kwargs["json"]["stream"] is True

    def test_returns_none_on_request_exception(self, adapter: OllamaAdapter) -> None:
        with patch("sensai.llm.ollama.requests.post", side_effect=requests.RequestException):
            assert adapter.chat({"model": "x", "messages": []}, stream=True) is None

    def test_returns_none_on_non_ok_status(self, adapter: OllamaAdapter) -> None:
        with patch("sensai.llm.ollama.requests.post", return_value=mock_response(ok=False)):
            assert adapter.chat({"model": "x", "messages": []}, stream=True) is None

    def test_closes_response_after_full_consumption(self, adapter: OllamaAdapter) -> None:
        lines = [json.dumps({"model": "m", "message": {"role": "assistant", "content": "hi"}, "done": True})]
        response = mock_response(lines=lines)
        with patch("sensai.llm.ollama.requests.post", return_value=response):
            result = adapter.chat({"model": "m", "messages": []}, stream=True)
            assert result is not None
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
            assert result is not None
            next(result)
            result.close()

        response.close.assert_called_once()

    def test_closes_response_on_read_error_mid_stream(self, adapter: OllamaAdapter) -> None:
        def raising_lines(*_args: object, **_kwargs: object) -> Iterator[str]:
            yield json.dumps({"model": "m", "message": {"role": "assistant", "content": "a"}, "done": False})
            raise requests.exceptions.ChunkedEncodingError

        response = mock_response(lines=[])
        response.iter_lines.side_effect = raising_lines
        with patch("sensai.llm.ollama.requests.post", return_value=response):
            result = adapter.chat({"model": "m", "messages": []}, stream=True)
            assert result is not None
            chunks = list(result)

        assert [chunk.content for chunk in chunks] == ["a"]
        response.close.assert_called_once()


class TestEmbedding:
    def test_calls_embed_endpoint_with_given_model(self, adapter: OllamaAdapter) -> None:
        with patch("sensai.llm.ollama.requests.post", return_value=mock_response(json_data={})) as mock_post:
            adapter.embeddings(["a"], "embeddinggemma")

        assert mock_post.call_args.args == (_API_PATH + "api/embed",)
        assert mock_post.call_args.kwargs["json"] == {"model": "embeddinggemma", "input": ["a"]}

    def test_embedding_returns_first_vector(self, adapter: OllamaAdapter) -> None:
        with patch(
            "sensai.llm.ollama.requests.post",
            return_value=mock_response(json_data={"embeddings": [[0.1, 0.2]]}),
        ):
            assert adapter.embedding("hello", "embeddinggemma") == [0.1, 0.2]

    def test_embedding_returns_empty_list_when_no_embeddings(self, adapter: OllamaAdapter) -> None:
        with patch("sensai.llm.ollama.requests.post", return_value=mock_response(json_data={"embeddings": []})):
            assert adapter.embedding("hello", "embeddinggemma") == []

    def test_embeddings_returns_vectors(self, adapter: OllamaAdapter) -> None:
        with patch(
            "sensai.llm.ollama.requests.post",
            return_value=mock_response(json_data={"embeddings": [[0.1], [0.2]]}),
        ):
            assert adapter.embeddings(["a", "b"], "embeddinggemma") == [[0.1], [0.2]]

    def test_embeddings_returns_empty_list_on_call_failure(self, adapter: OllamaAdapter) -> None:
        with patch("sensai.llm.ollama.requests.post", side_effect=requests.RequestException):
            assert adapter.embeddings(["a", "b"], "embeddinggemma") == []


class TestAvailableModels:
    def test_lists_installed_models(self, adapter: OllamaAdapter) -> None:
        tags = {"models": [{"name": "qwen2.5:1.5b"}, {"name": "llama3.2:latest"}]}
        with patch("sensai.llm.ollama.requests.get", return_value=mock_response(json_data=tags)) as mock_get:
            assert adapter.available_models() == ["qwen2.5:1.5b", "llama3.2:latest"]

        assert mock_get.call_args.args == (_API_PATH + "api/tags",)

    def test_returns_empty_list_when_no_model_is_installed(self, adapter: OllamaAdapter) -> None:
        with patch("sensai.llm.ollama.requests.get", return_value=mock_response(json_data={"models": []})):
            assert adapter.available_models() == []

    def test_returns_empty_list_on_call_failure(self, adapter: OllamaAdapter) -> None:
        with patch("sensai.llm.ollama.requests.get", side_effect=requests.RequestException):
            assert adapter.available_models() == []

    def test_returns_empty_list_on_non_ok_status(self, adapter: OllamaAdapter) -> None:
        with patch("sensai.llm.ollama.requests.get", return_value=mock_response(ok=False)):
            assert adapter.available_models() == []


class TestGetDefaultModel:
    def test_returns_first_installed_model(self, adapter: OllamaAdapter) -> None:
        tags = {"models": [{"name": "qwen2.5:1.5b"}, {"name": "llama3.2:latest"}]}
        with patch("sensai.llm.ollama.requests.get", return_value=mock_response(json_data=tags)):
            assert adapter.get_default_model() == "qwen2.5:1.5b"

    def test_returns_none_when_no_model_is_installed(self, adapter: OllamaAdapter) -> None:
        with patch("sensai.llm.ollama.requests.get", side_effect=requests.RequestException):
            assert adapter.get_default_model() is None


class TestShow:
    def test_returns_none_when_model_not_found(self, adapter: OllamaAdapter) -> None:
        with patch("sensai.llm.ollama.requests.post", return_value=mock_response(ok=False)):
            assert adapter.show("missing-model") is None

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

        assert result is not None
        assert result.tools is True
        assert result.think is True


class TestLoad:
    def test_returns_true_when_done(self, adapter: OllamaAdapter) -> None:
        with patch("sensai.llm.ollama.requests.post", return_value=mock_response(json_data={"done": True})):
            assert adapter.load("model") is True

    def test_returns_false_when_not_done(self, adapter: OllamaAdapter) -> None:
        with patch("sensai.llm.ollama.requests.post", return_value=mock_response(json_data={"done": False})):
            assert adapter.load("model") is False

    def test_returns_false_on_call_failure(self, adapter: OllamaAdapter) -> None:
        with patch("sensai.llm.ollama.requests.post", side_effect=requests.RequestException):
            assert adapter.load("model") is False


class TestUnload:
    def test_returns_true_when_done(self, adapter: OllamaAdapter) -> None:
        with patch(
            "sensai.llm.ollama.requests.post",
            return_value=mock_response(json_data={"done": True}),
        ) as mock_post:
            assert adapter.unload("model") is True
        assert mock_post.call_args.kwargs["json"]["keep_alive"] == 0

    def test_returns_false_on_call_failure(self, adapter: OllamaAdapter) -> None:
        with patch("sensai.llm.ollama.requests.post", side_effect=requests.RequestException):
            assert adapter.unload("model") is False
