from typing import cast

from fastapi import WebSocket
from sensai_protocol import ServerMessage, parse_server_message
from sensai_server.outbound import Outbound


class FakeCatalog:
    """A `ModelCatalog` serving the given models."""

    def __init__(self, models: list[str]) -> None:
        self.models = models

    async def available_models(self) -> list[str]:
        return list(self.models)


class RecordingOutbound(Outbound):
    """An `Outbound` that records the messages it sends instead of writing to a socket."""

    def __init__(self) -> None:
        super().__init__(cast("WebSocket", None))
        self.sent: list[ServerMessage] = []

    async def send(self, message: ServerMessage) -> None:
        self.sent.append(parse_server_message(message.model_dump_json()))
