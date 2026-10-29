from pathlib import Path
from uuid import uuid4

import pytest
from sensai_protocol import ClientCommand, CommandContext, FileContent, ReadFileParams, RequestCommand
from sensai_tui.client_commands import CLIENT_COMMAND_HANDLERS, ClientCommandFailedError, file, read_file
from sensai_tui.client_commands.dispatch import run_client_command

PERSONA_TOML = 'name = "Pirate"\ndescription = "Talks like a pirate"\nprompt = "Arr."\n'


def _request(params: dict) -> RequestCommand:
    return RequestCommand(request_id=uuid4(), command=ClientCommand.READ_FILE, params=params)


class TestReadFile:
    async def test_returns_the_raw_text(self, tmp_path: Path) -> None:
        path = tmp_path / "me.toml"
        path.write_text(PERSONA_TOML + "not = [valid", encoding="utf-8")
        assert await read_file(ReadFileParams(path=str(path))) == FileContent(content=PERSONA_TOML + "not = [valid")

    async def test_path_is_relative_to_the_current_directory(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        (tmp_path / "me.toml").write_text("x", encoding="utf-8")
        monkeypatch.chdir(tmp_path)
        assert await read_file(ReadFileParams(path="me.toml")) == FileContent(content="x")

    async def test_home_is_expanded(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        (tmp_path / "me.toml").write_text("x", encoding="utf-8")
        monkeypatch.setenv("HOME", str(tmp_path))
        assert await read_file(ReadFileParams(path="~/me.toml")) == FileContent(content="x")

    async def test_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(ClientCommandFailedError, match="file not found"):
            await read_file(ReadFileParams(path=str(tmp_path / "missing.toml")))

    async def test_directory(self, tmp_path: Path) -> None:
        with pytest.raises(ClientCommandFailedError, match="path is a directory"):
            await read_file(ReadFileParams(path=str(tmp_path)))

    async def test_not_utf8(self, tmp_path: Path) -> None:
        path = tmp_path / "me.toml"
        path.write_bytes(b"\xff\xfe\x00")
        with pytest.raises(ClientCommandFailedError, match="not valid UTF-8"):
            await read_file(ReadFileParams(path=str(path)))

    async def test_too_large(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(file, "MAX_FILE_SIZE", 4)
        path = tmp_path / "me.toml"
        path.write_text("12345", encoding="utf-8")
        with pytest.raises(ClientCommandFailedError, match="file too large"):
            await read_file(ReadFileParams(path=str(path)))


class TestRunClientCommand:
    async def test_answers_with_the_context(self, tmp_path: Path) -> None:
        path = tmp_path / "me.toml"
        path.write_text(PERSONA_TOML, encoding="utf-8")
        request = _request({"path": str(path)})
        assert await run_client_command(request) == CommandContext(
            request_id=request.request_id, command=ClientCommand.READ_FILE, ctx={"content": PERSONA_TOML}
        )

    async def test_failing_command_answers_with_its_reason(self, tmp_path: Path) -> None:
        request = _request({"path": str(tmp_path / "missing.toml")})
        assert await run_client_command(request) == CommandContext(
            request_id=request.request_id, command=ClientCommand.READ_FILE, error="file not found"
        )

    @pytest.mark.parametrize("params", [{}, {"path": ""}, {"path": "x", "extra": 1}])
    async def test_invalid_params_answer_with_an_error(self, params: dict) -> None:
        answer = await run_client_command(_request(params))
        assert answer.ctx is None
        assert answer.error

    async def test_unexpected_error_is_answered(self, monkeypatch: pytest.MonkeyPatch) -> None:
        async def broken(_: ReadFileParams) -> FileContent:
            msg = "boom"
            raise ValueError(msg)

        monkeypatch.setitem(CLIENT_COMMAND_HANDLERS, ClientCommand.READ_FILE, broken)
        answer = await run_client_command(_request({"path": "x"}))
        assert answer.error == "boom"

    async def test_missing_handler_is_answered(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delitem(CLIENT_COMMAND_HANDLERS, ClientCommand.READ_FILE)
        answer = await run_client_command(_request({"path": "x"}))
        assert answer.error == "the client does not implement 'read_file'"


class TestHandlersMap:
    def test_every_client_command_has_a_handler(self) -> None:
        assert set(CLIENT_COMMAND_HANDLERS) == set(ClientCommand)
