from __future__ import annotations

import threading
from collections import deque

from sensai.ui.adapter import Event, EventType


class EventQueue:
    """Thread-safe FIFO queue of UI events.

    Used to hand events (e.g. user input) from the thread running the UI
    to the worker thread consuming them, without either thread blocking
    the other beyond what's needed to acquire the lock.
    """

    def __init__(self) -> None:
        """Initialize an empty queue with its lock and wait condition."""
        self._queue = deque()
        self._lock = threading.Lock()
        self._not_empty = threading.Condition(self._lock)

    def put(self, content: str) -> None:
        """Add an event to the queue and wake any thread waiting in `wait_event`."""
        with self._not_empty:
            event = Event()
            event.content = content
            if event.content.startswith("/"):
                event.type = EventType.Command
            else:
                event.type = EventType.UserContent
            self._queue.append(event)
            self._not_empty.notify()

    def wait_event(self, *, timeout: float | None = None) -> bool:
        """Wait until there is at least one event, without consuming it."""
        with self._not_empty:
            if self._queue:
                return True
            return self._not_empty.wait_for(lambda: len(self._queue) > 0, timeout=timeout)

    def get_event(self) -> Event | None:
        """Get (and remove) the first event, if any."""
        with self._lock:
            if self._queue:
                return self._queue.popleft()
            return None
