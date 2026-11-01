import base64
from pathlib import Path
from uuid import uuid4

import pytest
from sensai_protocol import (
    BinaryFileContent,
    ClientCommand,
    CommandContext,
    FileContent,
    FileWritten,
    ReadFileParams,
    RequestCommand,
    WriteFileParams,
)
from sensai_tui.client_commands import (
    CLIENT_COMMAND_HANDLERS,
    ClientCommandFailedError,
    file,
    read_binary_file,
    read_file,
    write_file,
)
from sensai_tui.client_commands.dispatch import run_client_command

PERSONA_TOML = 'name = "Pirate"\ndescription = "Talks like a pirate"\nprompt = "Arr."\n'


def _request(params: dict, command: ClientCommand = ClientCommand.READ_FILE) -> RequestCommand:
    return RequestCommand(request_id=uuid4(), command=command, params=params)


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


class TestReadBinaryFile:
    async def test_returns_the_bytes_in_base64(self, tmp_path: Path) -> None:
        data = b"%PDF-1.7\n\xff\xfe\x00" + bytes(range(256))
        path = tmp_path / "doc.pdf"
        path.write_bytes(data)
        content = await read_binary_file(ReadFileParams(path=str(path)))
        assert isinstance(content, BinaryFileContent)
        assert base64.b64decode(content.content) == data

    async def test_empty_file(self, tmp_path: Path) -> None:
        path = tmp_path / "empty.pdf"
        path.write_bytes(b"")
        assert await read_binary_file(ReadFileParams(path=str(path))) == BinaryFileContent(content="")

    async def test_home_is_expanded(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        (tmp_path / "doc.pdf").write_bytes(b"x")
        monkeypatch.setenv("HOME", str(tmp_path))
        content = await read_binary_file(ReadFileParams(path="~/doc.pdf"))
        assert base64.b64decode(content.content) == b"x"

    async def test_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(ClientCommandFailedError, match="file not found"):
            await read_binary_file(ReadFileParams(path=str(tmp_path / "missing.pdf")))

    async def test_directory(self, tmp_path: Path) -> None:
        with pytest.raises(ClientCommandFailedError, match="path is a directory"):
            await read_binary_file(ReadFileParams(path=str(tmp_path)))

    async def test_too_large(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(file, "MAX_BINARY_FILE_SIZE", 4)
        path = tmp_path / "doc.pdf"
        path.write_bytes(b"12345")
        with pytest.raises(ClientCommandFailedError, match=r"file too large \(more than 4 bytes\)"):
            await read_binary_file(ReadFileParams(path=str(path)))

    async def test_text_limit_does_not_apply(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(file, "MAX_FILE_SIZE", 4)
        path = tmp_path / "doc.pdf"
        path.write_bytes(b"12345")
        content = await read_binary_file(ReadFileParams(path=str(path)))
        assert base64.b64decode(content.content) == b"12345"

    async def test_os_error(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        def denied(*_: object, **__: object) -> None:
            raise PermissionError(13, "Permission denied")

        path = tmp_path / "doc.pdf"
        path.write_bytes(b"x")
        monkeypatch.setattr(Path, "read_bytes", denied)
        with pytest.raises(ClientCommandFailedError, match=r"cannot read file \(Permission denied\)"):
            await read_binary_file(ReadFileParams(path=str(path)))


class TestWriteFile:
    async def test_writes_the_text_and_returns_the_absolute_path(self, tmp_path: Path) -> None:
        path = tmp_path / "branch.json"
        assert await write_file(WriteFileParams(path=str(path), content="{}\né")) == FileWritten(path=str(path))
        assert path.read_text(encoding="utf-8") == "{}\né"
        assert await read_file(ReadFileParams(path=str(path))) == FileContent(content="{}\né")

    async def test_path_is_relative_to_the_current_directory(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        written = await write_file(WriteFileParams(path="branch.json", content="x"))
        assert written == FileWritten(path=str((tmp_path / "branch.json").resolve()))
        assert (tmp_path / "branch.json").read_text(encoding="utf-8") == "x"

    async def test_home_is_expanded(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("HOME", str(tmp_path))
        await write_file(WriteFileParams(path="~/branch.json", content="x"))
        assert (tmp_path / "branch.json").read_text(encoding="utf-8") == "x"

    async def test_directory(self, tmp_path: Path) -> None:
        with pytest.raises(ClientCommandFailedError, match="path is a directory"):
            await write_file(WriteFileParams(path=str(tmp_path), content="x"))

    async def test_missing_directory(self, tmp_path: Path) -> None:
        with pytest.raises(ClientCommandFailedError, match="directory not found"):
            await write_file(WriteFileParams(path=str(tmp_path / "missing" / "branch.json"), content="x"))

    async def test_existing_file_is_overwritten(self, tmp_path: Path) -> None:
        path = tmp_path / "branch.json"
        path.write_text("an older and longer content", encoding="utf-8")
        assert await write_file(WriteFileParams(path=str(path), content="x")) == FileWritten(path=str(path))
        assert path.read_text(encoding="utf-8") == "x"

    async def test_os_error(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        def denied(*_: object, **__: object) -> None:
            raise PermissionError(13, "Permission denied")

        monkeypatch.setattr(Path, "write_text", denied)
        with pytest.raises(ClientCommandFailedError, match=r"cannot write file \(Permission denied\)"):
            await write_file(WriteFileParams(path=str(tmp_path / "branch.json"), content="x"))


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

    async def test_write_file_answers_with_the_context(self, tmp_path: Path) -> None:
        path = tmp_path / "branch.json"
        request = _request({"path": str(path), "content": "{}"}, ClientCommand.WRITE_FILE)
        assert await run_client_command(request) == CommandContext(
            request_id=request.request_id, command=ClientCommand.WRITE_FILE, ctx={"path": str(path)}
        )
        assert path.read_text(encoding="utf-8") == "{}"

    async def test_failing_write_file_answers_with_its_reason(self, tmp_path: Path) -> None:
        request = _request({"path": str(tmp_path), "content": "{}"}, ClientCommand.WRITE_FILE)
        assert await run_client_command(request) == CommandContext(
            request_id=request.request_id, command=ClientCommand.WRITE_FILE, error="path is a directory"
        )

    async def test_read_binary_file_answers_with_the_context(self, tmp_path: Path) -> None:
        path = tmp_path / "doc.pdf"
        path.write_bytes(b"\x00\xff")
        request = _request({"path": str(path)}, ClientCommand.READ_BINARY_FILE)
        assert await run_client_command(request) == CommandContext(
            request_id=request.request_id,
            command=ClientCommand.READ_BINARY_FILE,
            ctx={"content": base64.b64encode(b"\x00\xff").decode("ascii")},
        )

    async def test_failing_read_binary_file_answers_with_its_reason(self, tmp_path: Path) -> None:
        request = _request({"path": str(tmp_path / "missing.pdf")}, ClientCommand.READ_BINARY_FILE)
        assert await run_client_command(request) == CommandContext(
            request_id=request.request_id, command=ClientCommand.READ_BINARY_FILE, error="file not found"
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
