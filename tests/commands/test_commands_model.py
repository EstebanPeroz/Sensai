from unittest.mock import MagicMock

import pytest

from sensai.commands.model import Model
from tests.commands.fakes import FakeUI


@pytest.fixture
def provider_manager() -> MagicMock:
    return MagicMock()


@pytest.fixture
def refresh_context_usage() -> MagicMock:
    return MagicMock()


@pytest.fixture
def model(provider_manager: MagicMock, refresh_context_usage: MagicMock) -> Model:
    return Model(provider_manager, refresh_context_usage)


class TestModel:
    def test_no_arguments_displays_sorted_models(
        self, fake_ui: FakeUI, provider_manager: MagicMock, model: Model
    ) -> None:
        provider_manager.get_models.return_value = ["qwen3:1.7b", "llama3:8b"]
        provider_manager.provider.return_value.current_model.return_value = "qwen3:1.7b"

        model.execute()

        assert fake_ui.messages == ["Current model: qwen3:1.7b\nAvailable models:\n- llama3:8b\n- qwen3:1.7b"]

    def test_no_arguments_without_models(self, fake_ui: FakeUI, provider_manager: MagicMock, model: Model) -> None:
        provider_manager.get_models.return_value = []

        model.execute()

        assert fake_ui.messages == ["No models are available."]

    def test_one_argument_sets_model(self, fake_ui: FakeUI, provider_manager: MagicMock, model: Model) -> None:
        provider_manager.set_model.return_value = True

        model.execute("qwen2.5:1.5b")

        provider_manager.set_model.assert_called_once_with("qwen2.5:1.5b")
        assert fake_ui.messages == ["Setting model to 'qwen2.5:1.5b'...", "Model set to 'qwen2.5:1.5b'."]

    def test_setting_a_model_refreshes_the_context_usage(
        self, provider_manager: MagicMock, refresh_context_usage: MagicMock, model: Model
    ) -> None:
        provider_manager.set_model.return_value = True

        model.execute("qwen2.5:1.5b")

        refresh_context_usage.assert_called_once_with()

    def test_one_argument_with_unavailable_model(
        self, fake_ui: FakeUI, provider_manager: MagicMock, refresh_context_usage: MagicMock, model: Model
    ) -> None:
        provider_manager.set_model.return_value = False

        model.execute("missing")

        assert fake_ui.messages == ["Setting model to 'missing'...", "Model 'missing' is not available."]
        refresh_context_usage.assert_not_called()

    def test_too_many_arguments_displays_usage(
        self, fake_ui: FakeUI, provider_manager: MagicMock, model: Model
    ) -> None:
        model.execute("a", "b")

        provider_manager.set_model.assert_not_called()
        assert len(fake_ui.messages) == 1
        assert "Invalid number of arguments" in fake_ui.messages[0]
        assert "/model [model_name]" in fake_ui.messages[0]

    def test_completions_are_sorted_model_names(self, provider_manager: MagicMock, model: Model) -> None:
        provider_manager.get_models.return_value = ["qwen3:1.7b", "llama3:8b"]

        assert model.completions([]) == ["llama3:8b", "qwen3:1.7b"]
