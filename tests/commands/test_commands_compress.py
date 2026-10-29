from unittest.mock import MagicMock
from uuid import uuid4

from sensai.commands.compress import Compress
from sensai.llm.message import Message, Role
from sensai.memory.history import History
from sensai.pipeline.modules.compression import Compression, CompressionError, Compressor
from tests.commands.fakes import FakeUI


def conversation(*, started: bool = True) -> MagicMock:
    conversation = MagicMock()
    conversation.uuid = uuid4() if started else None
    conversation.history = History()
    conversation.history.insert(Message(uuid=None, role=Role.USER, content="Bonjour"))
    return conversation


class TestCompress:
    def test_summarises_the_current_branch(self, fake_ui: FakeUI) -> None:
        current, compressor = conversation(), MagicMock(spec=Compressor)
        compressor.compress.return_value = Compression(4, 900, 40)

        Compress(current, compressor).execute()

        compressor.compress.assert_called_once_with(current.uuid, [{"role": "user", "content": "Bonjour"}])
        assert fake_ui.messages == ["Summarised the 4 oldest messages of the conversation: 900 -> 40 tokens."]

    def test_nothing_left_to_summarise(self, fake_ui: FakeUI) -> None:
        compressor = MagicMock(spec=Compressor)
        compressor.compress.return_value = None

        Compress(conversation(), compressor).execute()

        assert fake_ui.messages == ["Nothing to compress: only the last messages are left to summarise."]

    def test_a_conversation_not_started(self, fake_ui: FakeUI) -> None:
        compressor = MagicMock(spec=Compressor)

        Compress(conversation(started=False), compressor).execute()

        compressor.compress.assert_not_called()
        assert fake_ui.messages == ["Nothing to compress: the conversation has not started."]

    def test_reports_a_failure(self, fake_ui: FakeUI) -> None:
        compressor = MagicMock(spec=Compressor)
        compressor.compress.side_effect = CompressionError("model not found")

        Compress(conversation(), compressor).execute()

        assert fake_ui.messages == ["The conversation could not be compressed: [Error]model not found"]

    def test_takes_no_argument(self, fake_ui: FakeUI) -> None:
        compressor = MagicMock(spec=Compressor)

        Compress(conversation(), compressor).execute("now")

        compressor.compress.assert_not_called()
        assert fake_ui.messages == ["This command takes no argument. Usage: /compress"]
