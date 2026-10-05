from pathlib import Path

import pytest

from sensai.config.settings import (
    PROVIDERS,
    CacheSettings,
    ConfigError,
    LLMSettings,
    OllamaSettings,
    Settings,
    load_settings,
)


def write_config(tmp_path: Path, content: str) -> Path:
    path = tmp_path / "settings.toml"
    path.write_text(content)
    return path


class TestPackagedSettings:
    def test_loads_packaged_file_by_default(self) -> None:
        settings = load_settings()

        assert isinstance(settings, Settings)
        assert settings.llm.embedding_model == "nomic-embed-text"
        assert settings.llm.providers == {"ollama": OllamaSettings()}
        assert settings.cache == CacheSettings()


class TestDefaults:
    def test_missing_keys_fall_back_to_defaults(self, tmp_path: Path) -> None:
        settings = load_settings(write_config(tmp_path, '[llm.ollama]\nbase_url = "http://ollama:1234"\n'))

        assert settings.llm.providers == {"ollama": OllamaSettings(base_url="http://ollama:1234")}
        assert settings.llm.embedding_model == LLMSettings().embedding_model

    def test_empty_file_gives_defaults(self, tmp_path: Path) -> None:
        assert load_settings(write_config(tmp_path, "")) == Settings()


class TestProviders:
    def test_no_provider_table_gives_no_provider(self, tmp_path: Path) -> None:
        assert load_settings(write_config(tmp_path, '[llm]\nembedding_model = "x"\n')).llm.providers == {}

    def test_empty_provider_table_uses_provider_defaults(self, tmp_path: Path) -> None:
        assert load_settings(write_config(tmp_path, "[llm.ollama]\n")).llm.providers == {"ollama": OllamaSettings()}

    def test_every_supported_provider_has_a_unique_name(self) -> None:
        names = [provider.name for provider in PROVIDERS]

        assert len(names) == len(set(names))


class TestEmbeddingModel:
    def test_is_read_from_llm_table(self, tmp_path: Path) -> None:
        settings = load_settings(write_config(tmp_path, '[llm]\nembedding_model = "embeddinggemma"\n'))

        assert settings.llm.embedding_model == "embeddinggemma"


class TestValidation:
    @pytest.mark.parametrize(
        ("content", "reason"),
        [
            ('[ollama]\nbase_url = "x"\n', "unexpected keyword argument 'ollama'"),
            ('[llm]\nmodel = "x"\n', "unexpected keyword argument 'model'"),
            ('[llm.ollama]\nurl = "x"\n', "unexpected keyword argument 'url'"),
            ('[llm.ollama]\nembedding_model = "x"\n', "unexpected keyword argument 'embedding_model'"),
            ('[llm.mistral]\napi_key = "x"\n', "unexpected keyword argument 'mistral'"),
            ('llm = "x"\n', "llm must be a table"),
            ('[llm]\nollama = "x"\n', "ollama must be a table"),
            ('[llm]\nembedding_model = ""\n', "llm.embedding_model must be a non-empty string"),
            ('[llm.ollama]\nbase_url = ["a"]\n', "llm.ollama.base_url must be a non-empty string"),
            ('[llm.ollama]\nbase_url = "http://"\n', "llm.ollama.base_url must be an http\\(s\\) URL with a host"),
            ('[llm.ollama]\nbase_url = "localhost:11434"\n', "llm.ollama.base_url must be an http\\(s\\) URL"),
            ('[llm.ollama]\nbase_url = "ftp://ollama:11434"\n', "llm.ollama.base_url must be an http\\(s\\) URL"),
            ('[llm.ollama]\nbase_url = "http://ollama:port"\n', "llm.ollama.base_url must be an http\\(s\\) URL"),
            ('cache = "x"\n', "cache must be a table"),
            ('[cache]\nhost = "x"\n', "unexpected keyword argument 'host'"),
            ('[cache]\nredis_url = ""\n', "cache.redis_url must be a non-empty string"),
            ('[cache]\nredis_url = "http://localhost:6379"\n', "cache.redis_url must be a redis://"),
            ('[cache]\nredis_url = "redis://"\n', "cache.redis_url must be a redis://"),
            ('[cache]\nredis_url = "redis://redis:port"\n', "cache.redis_url must be a redis://"),
            ('[cache]\nredis_url = "unix://"\n', "cache.redis_url must be a redis://"),
            ('[cache]\nsimilarity_threshold = "high"\n', "cache.similarity_threshold must be a number"),
            ("[cache]\nsimilarity_threshold = true\n", "cache.similarity_threshold must be a number"),
            ("[cache]\nsimilarity_threshold = 0\n", "cache.similarity_threshold must be in"),
            ("[cache]\nsimilarity_threshold = 1.5\n", "cache.similarity_threshold must be in"),
            ("[cache]\nttl = 1.5\n", "cache.ttl must be an integer"),
            ("[cache]\nttl = 0\n", "cache.ttl must be a positive number of seconds"),
        ],
    )
    def test_invalid_settings_raise_config_error(self, tmp_path: Path, content: str, reason: str) -> None:
        with pytest.raises(ConfigError, match=f"invalid settings in .*{reason}"):
            load_settings(write_config(tmp_path, content))


class TestBaseUrl:
    @pytest.mark.parametrize("url", ["http://localhost:11434/", "https://ollama.example.com", "http://127.0.0.1"])
    def test_accepts_http_urls(self, url: str) -> None:
        assert OllamaSettings(base_url=url).base_url == url


class TestCache:
    def test_is_read_from_cache_table(self, tmp_path: Path) -> None:
        content = '[cache]\nredis_url = "redis://cache:6380/1"\nsimilarity_threshold = 0.8\nttl = 60\n'

        settings = load_settings(write_config(tmp_path, content))

        assert settings.cache == CacheSettings(redis_url="redis://cache:6380/1", similarity_threshold=0.8, ttl=60)

    def test_missing_table_gives_defaults(self, tmp_path: Path) -> None:
        assert load_settings(write_config(tmp_path, "[llm]\n")).cache == CacheSettings()

    @pytest.mark.parametrize(
        "url", ["redis://localhost:6379/0", "rediss://user:pass@cache.example.com", "unix:///tmp/r.sock"]
    )
    def test_accepts_redis_urls(self, url: str) -> None:
        assert CacheSettings(redis_url=url).redis_url == url


class TestFileErrors:
    def test_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(ConfigError, match="config file not found"):
            load_settings(tmp_path / "nope.toml")

    def test_directory(self, tmp_path: Path) -> None:
        with pytest.raises(ConfigError, match="config path is a directory"):
            load_settings(tmp_path)

    def test_unreadable_file(self, tmp_path: Path) -> None:
        path = write_config(tmp_path, "")
        path.chmod(0)
        try:
            with pytest.raises(ConfigError, match="cannot read config file"):
                load_settings(path)
        finally:
            path.chmod(0o600)

    def test_not_utf8(self, tmp_path: Path) -> None:
        path = tmp_path / "settings.toml"
        path.write_bytes(b"\xff\xfe")

        with pytest.raises(ConfigError, match="config file is not valid UTF-8"):
            load_settings(path)

    def test_invalid_toml(self, tmp_path: Path) -> None:
        with pytest.raises(ConfigError, match="invalid TOML"):
            load_settings(write_config(tmp_path, "[llm\nembedding_model = \n"))
