from uuid import uuid4

import pydantic
import pytest
from sensai_protocol import (
    CLIENT_COMMANDS,
    MAX_FILE_SIZE,
    ClientCommand,
    CommandContext,
    FileContent,
    FileWritten,
    ProtocolError,
    ReadFileParams,
    WriteFileParams,
    parse_ctx,
    parse_params,
)


class TestClientCommandsMap:
    def test_every_command_has_its_models(self) -> None:
        assert set(CLIENT_COMMANDS) == set(ClientCommand)

    def test_read_file_models(self) -> None:
        assert CLIENT_COMMANDS[ClientCommand.READ_FILE] == (ReadFileParams, FileContent)

    def test_write_file_models(self) -> None:
        assert CLIENT_COMMANDS[ClientCommand.WRITE_FILE] == (WriteFileParams, FileWritten)


class TestParseParams:
    def test_valid_params(self) -> None:
        assert parse_params(ClientCommand.READ_FILE, {"path": "./me.toml"}) == ReadFileParams(path="./me.toml")

    @pytest.mark.parametrize("params", [{}, {"path": ""}, {"path": 1}, {"path": "x", "extra": 1}])
    def test_invalid_params(self, params: dict) -> None:
        with pytest.raises(ProtocolError):
            parse_params(ClientCommand.READ_FILE, params)

    def test_valid_write_file_params(self) -> None:
        params = {"path": "./branch.json", "content": "{}"}
        assert parse_params(ClientCommand.WRITE_FILE, params) == WriteFileParams(path="./branch.json", content="{}")

    def test_write_file_content_at_the_limit(self) -> None:
        assert parse_params(ClientCommand.WRITE_FILE, {"path": "x", "content": "x" * MAX_FILE_SIZE})

    @pytest.mark.parametrize(
        "params",
        [
            {},
            {"path": "x"},
            {"content": "x"},
            {"path": "", "content": "x"},
            {"path": "x", "content": 1},
            {"path": "x", "content": "x" * (MAX_FILE_SIZE + 1)},
            {"path": "x", "content": "x", "extra": 1},
        ],
        ids=["empty", "no-content", "no-path", "empty-path", "not-text", "too-large", "extra"],
    )
    def test_invalid_write_file_params(self, params: dict) -> None:
        with pytest.raises(ProtocolError):
            parse_params(ClientCommand.WRITE_FILE, params)


class TestParseCtx:
    def test_valid_ctx(self) -> None:
        assert parse_ctx(ClientCommand.READ_FILE, {"content": "raw"}) == FileContent(content="raw")

    def test_content_at_the_limit(self) -> None:
        assert parse_ctx(ClientCommand.READ_FILE, {"content": "x" * MAX_FILE_SIZE})

    @pytest.mark.parametrize(
        "ctx",
        [None, {}, {"content": 1}, {"content": "x" * (MAX_FILE_SIZE + 1)}],
        ids=["none", "missing", "not-text", "too-large"],
    )
    def test_invalid_ctx(self, ctx: dict | None) -> None:
        with pytest.raises(ProtocolError):
            parse_ctx(ClientCommand.READ_FILE, ctx)

    def test_valid_write_file_ctx(self) -> None:
        ctx = parse_ctx(ClientCommand.WRITE_FILE, {"path": "/home/me/branch.json"})
        assert ctx == FileWritten(path="/home/me/branch.json")
        assert parse_ctx(ClientCommand.WRITE_FILE, ctx.model_dump()) == ctx

    @pytest.mark.parametrize("ctx", [None, {}, {"path": ""}, {"path": 1}], ids=["none", "missing", "empty", "not-text"])
    def test_invalid_write_file_ctx(self, ctx: dict | None) -> None:
        with pytest.raises(ProtocolError):
            parse_ctx(ClientCommand.WRITE_FILE, ctx)


class TestCommandContext:
    @pytest.mark.parametrize(("ctx", "error"), [({"content": "x"}, "boom"), (None, None)])
    def test_needs_exactly_one_of_ctx_and_error(self, ctx: dict | None, error: str | None) -> None:
        with pytest.raises(pydantic.ValidationError):
            CommandContext(request_id=uuid4(), command=ClientCommand.READ_FILE, ctx=ctx, error=error)
