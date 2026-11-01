from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable


class EventBus:
    """Observer hub: emitters publish events without knowing who, if anyone, listens to them.

    Handlers are registered per event class and called synchronously, in subscription order,
    on the emitting thread. The server keeps one bus for the whole app, and it runs on the event
    loop that every session shares: handlers must be quick and must not block. A handler with I/O
    to do hands it off (to a thread, a background task) instead of doing it inline.

    A handler's exception propagates out of `emit`: the emitter decides how to contain it.
    """

    _handlers: defaultdict[type, list[Callable[[Any], None]]]

    def __init__(self) -> None:
        """Start with no subscriber."""
        self._handlers = defaultdict(list)

    def subscribe[E](self, event_type: type[E], handler: Callable[[E], None]) -> None:
        """Call `handler` with every event of type `event_type` (subclasses included) emitted from now on."""
        self._handlers[event_type].append(handler)

    def emit(self, event: object) -> None:
        """Pass `event` to every handler subscribed to its type or one of its base classes."""
        for event_type in type(event).__mro__:
            for handler in self._handlers.get(event_type, []):
                handler(event)
