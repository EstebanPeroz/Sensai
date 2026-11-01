import pytest
from sensai_protocol.error import SensaiError
from sensai_server.llm.error import EmbeddingError, JsonError, ModelError, RequestCallError, RequestStatusError


class TestErrors:
    @pytest.mark.parametrize(
        ("error", "text"),
        [
            (RequestCallError(" Get -> refused"), "[RequestCallError] Get -> refused"),
            (RequestStatusError(404), "[RequestStatusError] status 404"),
            (JsonError(), "[JsonError]"),
            (JsonError(" bad line"), "[JsonError] bad line"),
            (ModelError("no model"), "[ModelError] no model"),
            (EmbeddingError("no vector"), "[EmbeddingError] no vector"),
        ],
    )
    def test_message_names_the_error(self, error: SensaiError, text: str) -> None:
        assert str(error) == text
        assert error.message == "[Error]" + text

    @pytest.mark.parametrize(
        "error", [RequestCallError("x"), RequestStatusError(500), JsonError(), ModelError("x"), EmbeddingError("x")]
    )
    def test_is_the_protocol_sensai_error(self, error: SensaiError) -> None:
        """The provider manager catches `sensai_protocol`'s SensaiError: these errors must be one."""
        assert isinstance(error, SensaiError)
