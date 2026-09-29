#!/usr/bin/env python3

from __future__ import annotations

import threading
from typing import TYPE_CHECKING

from sensai.llm.ollama import OllamaAdapter
from sensai.ui.adapter import Event, EventType
from sensai.ui.tui.core import UITextualAdapter

if TYPE_CHECKING:
    from collections.abc import Iterator

    from sensai.llm.adapter import ProviderAdapter
    from sensai.llm.responses import ChatResponse
    from sensai.ui.adapter import UIAdapter


def _is_quit(event: Event | None) -> bool:
    return event is not None and event.type is EventType.UserContent and event.content == "/q"


def _stream_chunks(
    ui: UIAdapter,
    chunks: Iterator[ChatResponse],
    stop_streaming: threading.Event,
    stream_done: threading.Event,
) -> None:
    """Forward chunks to the UI until they run out or a stop is requested."""
    try:
        for chunk in chunks:
            if stop_streaming.is_set():
                return
            ui.send_ai_response(chunk)

            if chunk.error is not None:
                break
    finally:
        stream_done.set()


def run_chat(ui: UIAdapter, content: str, llm_adapter: ProviderAdapter, model: str) -> Event | None:
    """TMP."""
    ui.send_user_input(content)

    chunks = llm_adapter.chat(
        {"model": model, "messages": [{"role": "user", "content": content}]},
        stream=True,
    )
    if chunks is None:
        return None

    stop_streaming = threading.Event()
    stream_done = threading.Event()
    threading.Thread(
        target=_stream_chunks,
        args=(ui, chunks, stop_streaming, stream_done),
        daemon=True,
    ).start()

    while not stream_done.is_set():
        if not ui.wait_event(timeout=0.05):
            continue
        polled = ui.get_event()
        if polled is not None:
            stop_streaming.set()
            return polled
    return None


def _core_logic(ui: UIAdapter, llm_adapter: ProviderAdapter, model: str) -> None:
    """Run the chat loop on a background thread while the UI owns the main thread."""
    if ui.open() is False:
        return
    event: Event | None = None
    while not _is_quit(event):
        if not event:
            ui.wait_event()
            event = ui.get_event()
        if event is None:
            continue
        if _is_quit(event):
            break
        if event.type == EventType.UserContent:
            event = run_chat(ui, event.content, llm_adapter, model)
            continue
        event = None
    ui.close()


def main() -> None:
    """Entry point: run the program."""
    ui: UIAdapter = UITextualAdapter()
    llm_adapter: ProviderAdapter = OllamaAdapter()
    model = "qwen3:1.7b"

    response = llm_adapter.show(model)
    if response is None:
        print("model not found")
        return

    driver = threading.Thread(target=_core_logic, args=(ui, llm_adapter, model), daemon=True)
    driver.start()
    ui.run()


if __name__ == "__main__":
    main()
