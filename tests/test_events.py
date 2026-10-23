from sensai.core.events import EventBus


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
