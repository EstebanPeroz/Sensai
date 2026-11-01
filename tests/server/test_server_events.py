import pytest
from sensai_server.events import EventBus


class Ping:
    pass


class LoudPing(Ping):
    pass


class TestEventBus:
    def test_dispatches_by_type_in_subscription_order(self) -> None:
        bus, calls = EventBus(), []
        bus.subscribe(Ping, lambda _: calls.append("first"))
        bus.subscribe(Ping, lambda _: calls.append("second"))
        bus.subscribe(str, lambda _: calls.append("str"))

        bus.emit(Ping())

        assert calls == ["first", "second"]

    def test_base_class_handlers_receive_subclass_events(self) -> None:
        bus, base, sub = EventBus(), [], []
        bus.subscribe(Ping, base.append)
        bus.subscribe(LoudPing, sub.append)

        loud, plain = LoudPing(), Ping()
        bus.emit(loud)
        bus.emit(plain)

        assert base == [loud, plain]
        assert sub == [loud]

    def test_emit_without_subscriber_is_a_no_op(self) -> None:
        EventBus().emit(Ping())

    def test_handler_exception_propagates(self) -> None:
        bus = EventBus()

        def fail(_: Ping) -> None:
            message = "boom"
            raise RuntimeError(message)

        bus.subscribe(Ping, fail)

        with pytest.raises(RuntimeError, match="boom"):
            bus.emit(Ping())
