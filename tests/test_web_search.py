import logging
from unittest.mock import MagicMock, patch

import pytest
import requests
from ddgs.exceptions import RatelimitException

from sensai.config.settings import WebSettings
from sensai.error import SensaiError
from sensai.llm.embedder import Embedder
from sensai.llm.error import RequestCallError, RequestStatusError
from sensai.tools.web._ddgs import DuckDuckGoSearch
from sensai.tools.web._http import HttpFetch
from sensai.tools.web._ollama import OllamaWeb
from sensai.tools.web._shaping import chunk_words, clean_text, cosine, top_chunks, truncate_tokens
from sensai.tools.web._web_search import MAX_TOKENS, SEARCH_MAX_TOKENS, WebFetchTool, WebSearchTool, web_tools
from sensai.tools.web.backend import FetchBackend, FetchChain, SearchBackend, SearchChain
from sensai.tools.web.error import WebBackendError
from sensai.tools.web.results import Page, SearchResult

_HTML = """<html><head><title>Paris facts</title></head><body>
<nav>Home | About | Contact</nav>
<article><h1>Paris facts</h1>
<p>Paris is the capital of France. It has been the country's political centre for centuries and hosts most
national institutions, ministries and the parliament.</p>
<p>The city is crossed by the Seine and counts about two million inhabitants within its administrative limits,
while its metropolitan area is home to more than twelve million people.</p>
</article>
<footer>Copyright footer text</footer></body></html>"""


def mock_response(*, ok: bool = True, status_code: int = 200, json_data: object = None, text: str = "") -> MagicMock:
    response = MagicMock(spec=requests.Response)
    response.ok = ok
    response.status_code = status_code
    response.json.return_value = json_data
    response.text = text
    return response


class StubSearch(SearchBackend):
    def __init__(self, name: str, results: list[SearchResult] | None = None) -> None:
        self.name = name
        self._results = results
        self.calls = 0

    def search(self, query: str, max_results: int) -> list[SearchResult]:
        del query
        self.calls += 1
        if self._results is None:
            raise WebBackendError(self.name, "down")
        return self._results[:max_results]


class StubFetch(FetchBackend):
    def __init__(self, name: str, page: Page | None = None) -> None:
        self.name = name
        self._page = page

    def fetch(self, url: str) -> Page:
        del url
        if self._page is None:
            raise WebBackendError(self.name, "down")
        return self._page


class KeywordEmbedder(Embedder):
    """Embeds a text as [count of 'paris', 1]: closest to the query when it mentions paris as often."""

    def __init__(self) -> None:
        pass

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[float(text.lower().count("paris")), 1.0] for text in texts]


def make_results(count: int) -> list[SearchResult]:
    return [SearchResult(f"Title {i}", f"https://example.com/{i}", f"snippet {i}") for i in range(count)]


class TestOllamaWeb:
    def test_search_posts_query_with_bearer_key(self) -> None:
        backend = OllamaWeb("https://ollama.com", "secret", 5)
        payload = {"results": [{"title": "T", "url": "https://a.com", "content": "c"}]}
        with patch("sensai.tools.web._ollama.requests.post", return_value=mock_response(json_data=payload)) as post:
            results = backend.search("paris", 5)

        assert results == [SearchResult("T", "https://a.com", "c")]
        assert post.call_args.args == ("https://ollama.com/api/web_search",)
        assert post.call_args.kwargs["json"] == {"query": "paris", "max_results": 5}
        assert post.call_args.kwargs["headers"] == {"Authorization": "Bearer secret"}
        assert post.call_args.kwargs["timeout"] == 5

    def test_search_skips_results_without_url_and_caps_count(self) -> None:
        backend = OllamaWeb("https://ollama.com/", "secret", 5)
        payload = {"results": [{"title": "no url"}, *({"url": f"https://a.com/{i}"} for i in range(9))]}
        with patch("sensai.tools.web._ollama.requests.post", return_value=mock_response(json_data=payload)):
            results = backend.search("q", 3)

        assert [result.url for result in results] == ["https://a.com/0", "https://a.com/1"]

    def test_fetch_returns_page(self) -> None:
        backend = OllamaWeb("https://ollama.com/", "secret", 5)
        payload = {"title": "Paris", "content": "Paris is the capital.", "links": []}
        with patch("sensai.tools.web._ollama.requests.post", return_value=mock_response(json_data=payload)) as post:
            page = backend.fetch("https://a.com")

        assert page == Page("Paris", "https://a.com", "Paris is the capital.")
        assert post.call_args.args == ("https://ollama.com/api/web_fetch",)
        assert post.call_args.kwargs["json"] == {"url": "https://a.com"}

    @pytest.mark.parametrize("api_key", [None, ""])
    def test_missing_key_fails_without_request(self, api_key: str | None) -> None:
        backend = OllamaWeb("https://ollama.com/", api_key, 5)
        with (
            patch("sensai.tools.web._ollama.requests.post") as post,
            pytest.raises(WebBackendError, match="OLLAMA_API_KEY"),
        ):
            backend.search("q", 5)

        post.assert_not_called()

    @pytest.mark.parametrize(
        ("response", "reason"),
        [
            (requests.Timeout("timed out"), "timed out"),
            (requests.ConnectionError("refused"), "refused"),
            (mock_response(ok=False, status_code=401), "status 401"),
            (mock_response(json_data=["not", "a", "dict"]), "unexpected response shape"),
            (mock_response(json_data={}), "missing 'results'"),
        ],
    )
    def test_search_errors_raise_backend_error(self, response: object, reason: str) -> None:
        backend = OllamaWeb("https://ollama.com/", "secret", 5)
        kwargs = {"side_effect": response} if isinstance(response, Exception) else {"return_value": response}
        with patch("sensai.tools.web._ollama.requests.post", **kwargs), pytest.raises(WebBackendError, match=reason):
            backend.search("q", 5)

    def test_invalid_json_raises_backend_error(self) -> None:
        response = mock_response()
        response.json.side_effect = requests.exceptions.JSONDecodeError("Expecting value", "", 0)
        backend = OllamaWeb("https://ollama.com/", "secret", 5)
        with (
            patch("sensai.tools.web._ollama.requests.post", return_value=response),
            pytest.raises(WebBackendError, match="invalid JSON"),
        ):
            backend.fetch("https://a.com")

    def test_empty_page_raises_backend_error(self) -> None:
        backend = OllamaWeb("https://ollama.com/", "secret", 5)
        with (
            patch("sensai.tools.web._ollama.requests.post", return_value=mock_response(json_data={"content": " "})),
            pytest.raises(WebBackendError, match="empty page"),
        ):
            backend.fetch("https://a.com")


class TestDuckDuckGoSearch:
    def test_maps_ddgs_results(self) -> None:
        hits = [{"title": "T", "href": "https://a.com", "body": "b"}, {"title": "no href"}]
        with patch("sensai.tools.web._ddgs.DDGS") as ddgs:
            ddgs.return_value.text.return_value = hits
            results = DuckDuckGoSearch(7).search("paris", 5)

        assert results == [SearchResult("T", "https://a.com", "b")]
        ddgs.assert_called_once_with(timeout=7)
        ddgs.return_value.text.assert_called_once_with("paris", max_results=5)

    def test_rate_limit_raises_backend_error(self) -> None:
        with patch("sensai.tools.web._ddgs.DDGS") as ddgs:
            ddgs.return_value.text.side_effect = RatelimitException("202 Ratelimit")
            with pytest.raises(WebBackendError, match="ddgs: 202 Ratelimit"):
                DuckDuckGoSearch(7).search("paris", 5)


class TestHttpFetch:
    def test_extracts_main_content_and_title(self) -> None:
        with patch("sensai.tools.web._http.requests.get", return_value=mock_response(text=_HTML)) as get:
            page = HttpFetch(4).fetch("https://a.com/paris")

        assert page.title == "Paris facts"
        assert page.url == "https://a.com/paris"
        assert "capital of France" in page.text
        assert "Copyright footer" not in page.text
        assert "Home | About" not in page.text
        assert get.call_args.kwargs["timeout"] == 4

    @pytest.mark.parametrize(
        ("response", "reason"),
        [
            (requests.Timeout("timed out"), "timed out"),
            (mock_response(ok=False, status_code=404), "status 404"),
            (mock_response(text="<html><body></body></html>"), "no main content"),
        ],
    )
    def test_errors_raise_backend_error(self, response: object, reason: str) -> None:
        kwargs = {"side_effect": response} if isinstance(response, Exception) else {"return_value": response}
        with patch("sensai.tools.web._http.requests.get", **kwargs), pytest.raises(WebBackendError, match=reason):
            HttpFetch(4).fetch("https://a.com")


class TestChain:
    def test_search_uses_first_backend_that_answers(self, caplog: pytest.LogCaptureFixture) -> None:
        first, second, third = StubSearch("a"), StubSearch("b", make_results(1)), StubSearch("c", make_results(2))
        with caplog.at_level(logging.INFO, logger="sensai.tools.web.backend"):
            results = SearchChain([first, second, third]).search("q", 5)

        assert results == make_results(1)
        assert third.calls == 0
        assert "a failed" in caplog.text
        assert "answered by b" in caplog.text

    def test_search_raises_when_every_backend_fails(self) -> None:
        with pytest.raises(WebBackendError, match=r"chain: a: down; b: down"):
            SearchChain([StubSearch("a"), StubSearch("b")]).search("q", 5)

    def test_fetch_uses_first_backend_that_answers(self, caplog: pytest.LogCaptureFixture) -> None:
        page = Page("T", "https://a.com", "text")
        with caplog.at_level(logging.INFO, logger="sensai.tools.web.backend"):
            assert FetchChain([StubFetch("a"), StubFetch("b", page)]).fetch("https://a.com") == page

        assert "answered by b" in caplog.text

    def test_fetch_raises_when_every_backend_fails(self) -> None:
        with pytest.raises(WebBackendError, match="a: down"):
            FetchChain([StubFetch("a")]).fetch("https://a.com")

    def test_backend_error_is_a_sensai_error(self) -> None:
        assert issubclass(WebBackendError, SensaiError)


class TestShaping:
    def test_chunk_words_splits_by_word_count(self) -> None:
        assert chunk_words("a b c d e", 2) == ["a b", "c d", "e"]

    def test_chunk_words_of_empty_text(self) -> None:
        assert chunk_words("  ", 2) == []

    def test_cosine(self) -> None:
        assert cosine([1.0, 0.0], [2.0, 0.0]) == pytest.approx(1.0)
        assert cosine([1.0, 0.0], [0.0, 3.0]) == pytest.approx(0.0)
        assert cosine([0.0, 0.0], [1.0, 1.0]) == 0.0

    def test_top_chunks_keeps_best_in_original_order(self) -> None:
        chunks = ["paris paris", "london", "paris", "berlin", "paris paris paris"]

        assert top_chunks(chunks, "paris paris paris", KeywordEmbedder(), 2) == ["paris paris", "paris paris paris"]

    def test_top_chunks_without_embeddings_keeps_first_chunks(self) -> None:
        embedder = MagicMock(spec=Embedder)
        embedder.embed.return_value = []

        assert top_chunks(["a", "b", "c"], "q", embedder, 2) == ["a", "b"]

    def test_top_chunks_skips_embedding_when_few_chunks(self) -> None:
        embedder = MagicMock(spec=Embedder)

        assert top_chunks(["a", "b"], "q", embedder, 4) == ["a", "b"]
        embedder.embed.assert_not_called()

    def test_clean_text_drops_link_targets_images_and_urls(self) -> None:
        text = "See ![logo](https://a.com/x.png) the [docs](https://a.com/docs)\n\n  at https://a.com/page now."

        assert clean_text(text) == "See the docs at now."

    def test_truncate_tokens(self) -> None:
        assert truncate_tokens("short", 10) == "short"
        truncated = truncate_tokens("x" * 100, 10)
        assert len(truncated) == 40
        assert truncated.endswith("[…]")


class TestEmbedder:
    def test_embeds_with_configured_model(self) -> None:
        provider = MagicMock()
        provider.embeddings.return_value = [[1.0], [2.0]]

        assert Embedder(provider, "nomic-embed-text").embed(["a", "b"]) == [[1.0], [2.0]]
        provider.embeddings.assert_called_once_with(["a", "b"], "nomic-embed-text")

    def test_mismatched_count_gives_no_vectors(self) -> None:
        provider = MagicMock()
        provider.embeddings.return_value = []

        assert Embedder(provider, "m").embed(["a"]) == []

    @pytest.mark.parametrize("error", [RequestCallError(" Post -> timed out"), RequestStatusError(500)])
    def test_provider_error_gives_no_vectors(self, error: SensaiError) -> None:
        provider = MagicMock()
        provider.embeddings.side_effect = error

        assert Embedder(provider, "m").embed(["a"]) == []

    def test_empty_input_skips_provider(self) -> None:
        provider = MagicMock()

        assert Embedder(provider, "m").embed([]) == []
        provider.embeddings.assert_not_called()


class TestWebSearchTool:
    def test_formats_numbered_results(self) -> None:
        tool = WebSearchTool(StubSearch("a", make_results(2)))

        assert tool.run(query="q") == (
            "[1] Title 0 | https://example.com/0\nsnippet 0\n\n[2] Title 1 | https://example.com/1\nsnippet 1"
        )

    def test_returns_at_most_five_results(self) -> None:
        output = WebSearchTool(StubSearch("a", make_results(9))).run(query="q")

        assert "[5]" in output
        assert "[6]" not in output

    def test_long_snippets_are_cut(self) -> None:
        backend = StubSearch("a", [SearchResult("T", "https://a.com", "word " * 500)])

        assert len(WebSearchTool(backend).run(query="q")) < 400

    def test_whole_output_is_capped(self) -> None:
        long_url = "https://example.com/" + "a" * 2000
        backend = StubSearch("a", [SearchResult("T", long_url, "s") for _ in range(5)])
        output = WebSearchTool(backend).run(query="q")

        assert len(output) <= SEARCH_MAX_TOKENS * 4
        assert output.endswith("[…]")

    def test_untitled_result_shows_url(self) -> None:
        backend = StubSearch("a", [SearchResult("", "https://a.com", "s")])

        assert WebSearchTool(backend).run(query="q").startswith("[1] https://a.com | https://a.com")

    def test_no_results(self) -> None:
        assert WebSearchTool(StubSearch("a", [])).run(query="paris") == "No web results for: paris"

    def test_failure_is_reported_as_unavailable(self) -> None:
        output = WebSearchTool(StubSearch("a")).run(query="q")

        assert output.startswith("Web search unavailable (down)")
        assert "could not be verified" in output

    def test_requires_query(self) -> None:
        with pytest.raises(SensaiError, match="Miss parameter: query"):
            WebSearchTool(StubSearch("a")).run()

    def test_tool_rule(self) -> None:
        rule = WebSearchTool(StubSearch("a")).tool_rule

        assert rule["function"]["name"] == "web_search"
        assert rule["function"]["parameters"]["required"] == ["query"]


class TestWebFetchTool:
    def test_returns_header_and_relevant_passages_in_order(self) -> None:
        words = [f"w{i}" for i in range(600)]
        words[10] = "paris"
        words[500] = "paris"
        page = Page("Paris", "https://a.com", " ".join(words))
        output = WebFetchTool(StubFetch("a", page), KeywordEmbedder()).run(url="https://a.com", query="paris")

        header, *passages = output.split("\n\n")
        assert header == "[1] Paris | https://a.com"
        assert len(passages) == 4
        assert passages[0].startswith("w0 ")
        assert "paris" in passages[0]
        assert any("paris" in passage for passage in passages[1:])

    def test_result_is_capped(self) -> None:
        page = Page("T", "https://a.com", f"{'x' * 20} " * 5000)
        output = WebFetchTool(StubFetch("a", page), KeywordEmbedder()).run(url="https://a.com", query="q")

        assert len(output) <= MAX_TOKENS * 4
        assert output.endswith("[…]")

    def test_embedding_failure_keeps_first_passages(self) -> None:
        provider = MagicMock()
        provider.embeddings.side_effect = RequestCallError(" Post -> timed out")
        page = Page("T", "https://a.com", " ".join(f"w{i}" for i in range(1000)))
        output = WebFetchTool(StubFetch("a", page), Embedder(provider, "m")).run(url="https://a.com", query="q")

        _, *passages = output.split("\n\n")
        assert passages[0].startswith("w0 ")
        assert passages[-1].startswith("w450 ")

    def test_failure_is_reported_as_unavailable(self) -> None:
        output = WebFetchTool(StubFetch("a"), KeywordEmbedder()).run(url="https://a.com", query="q")

        assert output.startswith("Web fetch unavailable")

    def test_requires_url_and_query(self) -> None:
        tool = WebFetchTool(StubFetch("a"), KeywordEmbedder())
        with pytest.raises(SensaiError, match="Miss parameter: query"):
            tool.run(url="https://a.com")

    def test_tool_rule(self) -> None:
        rule = WebFetchTool(StubFetch("a"), KeywordEmbedder()).tool_rule

        assert rule["function"]["name"] == "web_fetch"
        assert rule["function"]["parameters"]["required"] == ["url", "query"]


class TestWebTools:
    def test_falls_back_to_keyless_backends_without_api_key(self) -> None:
        search, fetch = web_tools(WebSettings(), None, KeywordEmbedder())
        with patch("sensai.tools.web._ddgs.DDGS") as ddgs, patch("sensai.tools.web._ollama.requests.post") as post:
            ddgs.return_value.text.return_value = [{"title": "T", "href": "https://a.com", "body": "b"}]
            assert search.run(query="q") == "[1] T | https://a.com\nb"
        post.assert_not_called()

        with patch("sensai.tools.web._http.requests.get", return_value=mock_response(text=_HTML)):
            assert fetch.run(url="https://a.com", query="paris").startswith("[1] Paris facts | https://a.com")

    def test_uses_ollama_with_api_key(self) -> None:
        search, _ = web_tools(WebSettings(api_base_url="https://ollama.test"), "key", KeywordEmbedder())
        payload = {"results": [{"title": "T", "url": "https://a.com", "content": "c"}]}
        with (
            patch("sensai.tools.web._ollama.requests.post", return_value=mock_response(json_data=payload)) as post,
            patch("sensai.tools.web._ddgs.DDGS") as ddgs,
        ):
            assert search.run(query="q") == "[1] T | https://a.com\nc"

        assert post.call_args.args == ("https://ollama.test/api/web_search",)
        ddgs.assert_not_called()
