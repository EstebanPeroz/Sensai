from sensai_server.models import DEFAULT_MODELS, StaticModelCatalog


class TestStaticModelCatalog:
    async def test_serves_default_models(self) -> None:
        assert await StaticModelCatalog().available_models() == DEFAULT_MODELS

    async def test_returns_a_copy(self) -> None:
        catalog = StaticModelCatalog()
        (await catalog.available_models()).append("other")
        assert await catalog.available_models() == DEFAULT_MODELS
