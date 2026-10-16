from unittest.mock import MagicMock
from uuid import UUID, uuid4

import pytest

from sensai.commands.branch import Branch
from sensai.memory.persistent_db.error import DatabaseError, InvalidInstanceError
from tests.commands.fakes import FakeUI

CURRENT = uuid4()
OTHER = uuid4()
INVALID_UUID = "parameter is not a valid UUID."


@pytest.fixture
def db(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    database = MagicMock()
    monkeypatch.setattr("sensai.commands.branch.get_db", lambda: database)
    return database


@pytest.fixture
def conv() -> MagicMock:
    conversation = MagicMock()
    conversation.uuid = CURRENT
    return conversation


@pytest.fixture
def provider_manager() -> MagicMock:
    return MagicMock()


@pytest.fixture
def load_conversation(fake_ui: FakeUI) -> MagicMock:
    fake_ui.load_conversation = MagicMock()  # type: ignore[method-assign]
    return fake_ui.load_conversation  # type: ignore[return-value]


@pytest.fixture
def branch(fake_ui: FakeUI, conv: MagicMock, provider_manager: MagicMock, db: MagicMock) -> Branch:  # noqa: ARG001
    return Branch(fake_ui, conv, provider_manager)


class TestInit:
    def test_without_database_raises(
        self,
        monkeypatch: pytest.MonkeyPatch,
        fake_ui: FakeUI,
        conv: MagicMock,
        provider_manager: MagicMock,
    ) -> None:
        monkeypatch.setattr("sensai.commands.branch.get_db", lambda: None)

        with pytest.raises(DatabaseError):
            Branch(fake_ui, conv, provider_manager)


class TestDispatch:
    def test_no_arguments_reports_missing_subcommand(self, branch: Branch, fake_ui: FakeUI) -> None:
        branch.execute()

        assert len(fake_ui.messages) == 1
        assert "Missing subcommand" in fake_ui.messages[0]
        assert branch.usage in fake_ui.messages[0]

    def test_unknown_subcommand_is_reported(self, branch: Branch, fake_ui: FakeUI, db: MagicMock) -> None:
        branch.execute("explode")

        assert len(fake_ui.messages) == 1
        assert "Unknown subcommand 'explode'" in fake_ui.messages[0]
        assert branch.usage in fake_ui.messages[0]
        db.get_branches.assert_not_called()

    @pytest.mark.parametrize(
        "args",
        [
            ("list", "extra"),
            ("set",),
            ("set", str(OTHER), "extra"),
            ("remove",),
            ("remove", str(OTHER), "extra"),
            ("rename",),
            ("rename", str(OTHER)),
            ("rename", str(OTHER), "name", "extra"),
            ("fork", "extra"),
        ],
    )
    def test_wrong_number_of_arguments_reports_usage(
        self, branch: Branch, fake_ui: FakeUI, args: tuple[str, ...]
    ) -> None:
        branch.execute(*args)

        assert len(fake_ui.messages) == 1
        assert f"Invalid arguments for '{args[0]}'" in fake_ui.messages[0]
        assert branch.usage in fake_ui.messages[0]

    def test_completions_are_the_sorted_subcommands(self, branch: Branch) -> None:
        assert branch.completions([]) == ["fork", "list", "remove", "rename", "set"]


class TestList:
    def test_displays_each_branch_with_its_uuid(self, branch: Branch, fake_ui: FakeUI, db: MagicMock) -> None:
        db.get_branches.return_value = {CURRENT: "Branch", OTHER: "Branch 2"}

        branch.execute("list")

        assert fake_ui.messages == [f"{CURRENT}: Branch", f"{OTHER}: Branch 2"]

    def test_without_branches_displays_nothing(self, branch: Branch, fake_ui: FakeUI, db: MagicMock) -> None:
        db.get_branches.return_value = {}

        branch.execute("list")

        assert fake_ui.messages == []


class TestSet:
    def test_loads_the_branch_into_the_conversation(  # noqa: PLR0913, PLR0917
        self,
        branch: Branch,
        fake_ui: FakeUI,
        db: MagicMock,
        conv: MagicMock,
        provider_manager: MagicMock,
        load_conversation: MagicMock,
    ) -> None:
        branch.execute("set", str(OTHER))

        db.get_branch.assert_called_once_with(current=conv, target_branch=OTHER, provider_manager=provider_manager)
        load_conversation.assert_called_once_with(conv.history)
        assert fake_ui.messages == []

    def test_invalid_uuid_is_reported(
        self,
        branch: Branch,
        fake_ui: FakeUI,
        db: MagicMock,
        load_conversation: MagicMock,
    ) -> None:
        branch.execute("set", "not-a-uuid")

        assert fake_ui.messages == [INVALID_UUID]
        db.get_branch.assert_not_called()
        load_conversation.assert_not_called()

    def test_unknown_branch_is_reported_and_nothing_is_loaded(
        self,
        branch: Branch,
        fake_ui: FakeUI,
        db: MagicMock,
        load_conversation: MagicMock,
    ) -> None:
        db.get_branch.side_effect = InvalidInstanceError("given uuid is invalid")

        branch.execute("set", str(OTHER))

        assert fake_ui.messages == ["given uuid is invalid"]
        load_conversation.assert_not_called()


class TestRemove:
    def test_removes_the_branch(self, branch: Branch, fake_ui: FakeUI, db: MagicMock) -> None:
        branch.execute("remove", str(OTHER))

        db.remove_branch.assert_called_once_with(OTHER)
        assert fake_ui.messages == [f"Branch {OTHER} removed"]

    def test_invalid_uuid_is_reported(self, branch: Branch, fake_ui: FakeUI, db: MagicMock) -> None:
        branch.execute("remove", "not-a-uuid")

        assert fake_ui.messages == [INVALID_UUID]
        db.remove_branch.assert_not_called()

    def test_current_branch_cannot_be_removed(self, branch: Branch, fake_ui: FakeUI, db: MagicMock) -> None:
        branch.execute("remove", str(CURRENT))

        assert fake_ui.messages == ["You cannot remove your current branch."]
        db.remove_branch.assert_not_called()

    def test_unknown_branch_is_reported(self, branch: Branch, fake_ui: FakeUI, db: MagicMock) -> None:
        db.remove_branch.side_effect = InvalidInstanceError("given uuid is invalid")

        branch.execute("remove", str(OTHER))

        assert fake_ui.messages == [f"Failed to remove {OTHER}"]


class TestRename:
    def test_renames_the_branch(self, branch: Branch, fake_ui: FakeUI, db: MagicMock) -> None:
        branch.execute("rename", str(OTHER), "work")

        db.set_branch_name.assert_called_once_with(OTHER, "work")
        assert fake_ui.messages == [f"Branch {OTHER} renamed work"]

    def test_invalid_uuid_is_reported(self, branch: Branch, fake_ui: FakeUI, db: MagicMock) -> None:
        branch.execute("rename", "not-a-uuid", "work")

        assert fake_ui.messages == [INVALID_UUID]
        db.set_branch_name.assert_not_called()

    def test_unknown_branch_is_reported(self, branch: Branch, fake_ui: FakeUI, db: MagicMock) -> None:
        db.set_branch_name.side_effect = InvalidInstanceError("given uuid is invalid")

        branch.execute("rename", str(OTHER), "work")

        assert fake_ui.messages == [f"Failed to rename {OTHER}"]


class TestDerive:
    def test_creates_a_copy_of_the_current_branch_and_loads_it(  # noqa: PLR0913, PLR0917
        self,
        branch: Branch,
        fake_ui: FakeUI,
        db: MagicMock,
        conv: MagicMock,
        provider_manager: MagicMock,
        load_conversation: MagicMock,
    ) -> None:
        db.create_branch.return_value = OTHER

        branch.execute("fork")

        db.create_branch.assert_called_once_with(CURRENT)
        db.get_branch.assert_called_once_with(current=conv, target_branch=OTHER, provider_manager=provider_manager)
        load_conversation.assert_called_once_with(conv.history)
        assert fake_ui.messages == []

    def test_without_a_started_conversation_is_reported(
        self,
        branch: Branch,
        fake_ui: FakeUI,
        db: MagicMock,
        conv: MagicMock,
    ) -> None:
        conv.uuid = None

        branch.execute("fork")

        assert fake_ui.messages == ["You cannot fork without a conversation started/setted"]
        db.create_branch.assert_not_called()

    def test_failed_creation_is_reported_and_nothing_is_loaded(
        self,
        branch: Branch,
        fake_ui: FakeUI,
        db: MagicMock,
        load_conversation: MagicMock,
    ) -> None:
        db.create_branch.return_value = None

        branch.execute("fork")

        assert fake_ui.messages == ["Failed to make a new branch"]
        db.get_branch.assert_not_called()
        load_conversation.assert_not_called()


def test_try_parse_uuid() -> None:
    value = uuid4()

    assert Branch._try_parse_uuid(str(value)) == value  # noqa: SLF001
    assert Branch._try_parse_uuid("nope") is None  # noqa: SLF001
    assert isinstance(Branch._try_parse_uuid(str(value)), UUID)  # noqa: SLF001
