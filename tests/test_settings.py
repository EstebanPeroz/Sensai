from importlib import resources
from pathlib import Path

import pytest

from sensai.config.settings import (
    PROVIDERS,
    CacheSettings,
    ConfigError,
    ContextSettings,
    FeatureSettings,
    LLMSettings,
    LoggingSettings,
    OllamaSettings,
    PersonaSettings,
    RagDBSettings,
    Settings,
    WebSettings,
    default_settings_file,
    load_settings,
)


def write_config(tmp_path: Path, content: str) -> Path:
    path = tmp_path / "settings.toml"
    path.write_text(content)
    return path


class TestPackagedSettings:
    def test_loads_default_file_without_path(self) -> None:
        assert isinstance(load_settings(), Settings)

    def test_template_holds_the_defaults(self) -> None:
        template = resources.files("sensai.config").joinpath("settings.toml.template")
        settings = load_settings(Path(str(template)))

        assert settings.llm.embedding_model == "nomic-embed-text:latest"
        assert settings.llm.providers == {"ollama": OllamaSettings()}
        assert settings.cache == CacheSettings()
        assert settings.db.rag.location == "http://localhost:6333"
        assert settings.web == WebSettings()
        assert settings.logging == LoggingSettings(enabled=True, log_content=True)
        assert settings.context == ContextSettings(budget=8192, warning_threshold=0.8, keep_recent=6)

    def test_default_file_is_the_template_without_local_settings(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        (tmp_path / "settings.toml.template").write_text("")
        monkeypatch.setattr("sensai.config.settings.resources.files", lambda _: tmp_path)

        assert default_settings_file() == tmp_path / "settings.toml.template"

    def test_default_file_is_the_local_settings_when_present(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        (tmp_path / "settings.toml.template").write_text("")
        (tmp_path / "settings.toml").write_text("")
        monkeypatch.setattr("sensai.config.settings.resources.files", lambda _: tmp_path)

        assert default_settings_file() == tmp_path / "settings.toml"


class TestDefaults:
    def test_missing_keys_fall_back_to_defaults(self, tmp_path: Path) -> None:
        settings = load_settings(write_config(tmp_path, '[llm.ollama]\nbase_url = "http://ollama:1234"\n'))

        assert settings.llm.providers == {"ollama": OllamaSettings(base_url="http://ollama:1234")}
        assert settings.llm.embedding_model == LLMSettings().embedding_model

    def test_empty_file_gives_defaults(self, tmp_path: Path) -> None:
        assert load_settings(write_config(tmp_path, "")) == Settings()

    def test_missing_mcp_configuration_gives_no_mcp_server(self, tmp_path: Path) -> None:
        assert load_settings(write_config(tmp_path, "")).mcp is None


class TestMCP:
    def test_loads_one_configured_server(self, tmp_path: Path) -> None:
        settings = load_settings(
            write_config(
                tmp_path,
                '[mcp]\ncommand = "uv"\nargs = ["run", "server.py"]\nstartup_timeout = 5\ncall_timeout = 20\n',
            ),
        )

        assert settings.mcp is not None
        assert settings.mcp.command == "uv"
        assert settings.mcp.args == ["run", "server.py"]
        assert settings.mcp.enabled is True
        assert settings.mcp.startup_timeout == 5
        assert settings.mcp.call_timeout == 20

    def test_loads_disabled_server(self, tmp_path: Path) -> None:
        settings = load_settings(write_config(tmp_path, '[mcp]\ncommand = "server"\nenabled = false\n'))

        assert settings.mcp is not None
        assert settings.mcp.enabled is False

    @pytest.mark.parametrize(
        ("content", "reason"),
        [
            ('[mcp]\ncommand = ""\n', "command must be a non-empty string"),
            ('[mcp]\ncommand = "server"\nstartup_timeout = 0\n', "startup timeout must be greater than zero"),
            ('[mcp]\ncommand = "server"\ncall_timeout = -1\n', "call timeout must be greater than zero"),
        ],
    )
    def test_invalid_mcp_values_raise_config_error(self, tmp_path: Path, content: str, reason: str) -> None:
        with pytest.raises(ConfigError, match=f"invalid settings in .*{reason}"):
            load_settings(write_config(tmp_path, content))


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


class TestRagDatabase:
    def test_is_read_from_the_db_table(self, tmp_path: Path) -> None:
        settings = load_settings(write_config(tmp_path, '[db]\nrag.location = "http://qdrant:6333"\n'))

        assert settings.db.rag.location == "http://qdrant:6333"

    def test_a_missing_table_is_not_an_error(self, tmp_path: Path) -> None:
        assert load_settings(write_config(tmp_path, "")).db.rag == RagDBSettings()

    def test_the_default_is_a_usable_url(self) -> None:
        assert RagDBSettings().location == "http://localhost:6333"

    def test_it_does_not_disturb_the_persistent_database(self, tmp_path: Path) -> None:
        content = '[db]\npersistent.url = "sqlite:///main.db"\nrag.location = "chunks_qdrant"\n'

        settings = load_settings(write_config(tmp_path, content))

        assert settings.db.persistent.url == "sqlite:///main.db"
        assert settings.db.rag.location == "chunks_qdrant"


class TestWeb:
    def test_is_read_from_web_table(self, tmp_path: Path) -> None:
        content = '[web]\napi_base_url = "https://example.com"\napi_key = "secret"\ntimeout = 3\n'
        settings = load_settings(write_config(tmp_path, content))

        assert settings.web == WebSettings(api_base_url="https://example.com", api_key="secret", timeout=3)

    def test_api_key_is_hidden_from_repr(self) -> None:
        assert "secret" not in repr(WebSettings(api_key="secret"))


class TestFeatures:
    def test_confirm_is_read_from_features_table(self, tmp_path: Path) -> None:
        settings = load_settings(write_config(tmp_path, '[features]\nconfirm = ["tools.web_fetch"]\n'))

        assert settings.features.confirm == ["tools.web_fetch"]

    def test_confirm_defaults_to_empty(self, tmp_path: Path) -> None:
        assert load_settings(write_config(tmp_path, "")).features.confirm == []

    def test_confirm_in_strips_the_namespace(self) -> None:
        features = FeatureSettings(confirm=["tools.web_fetch", "commands.export", "tools.ask_user"])

        assert features.confirm_in("tools") == {"web_fetch", "ask_user"}

    def test_confirm_and_disabled_are_independent(self) -> None:
        features = FeatureSettings(disabled=["tools.a"], confirm=["tools.b"])

        assert features.disabled_in("tools") == {"a"}
        assert features.confirm_in("tools") == {"b"}


class TestPersonas:
    def test_security_is_read_from_personas_table(self, tmp_path: Path) -> None:
        settings = load_settings(write_config(tmp_path, '[personas]\nsecurity = ["a.toml", "b.toml"]\n'))

        assert settings.personas.security == ["a.toml", "b.toml"]

    def test_security_defaults_to_the_packaged_persona(self, tmp_path: Path) -> None:
        settings = load_settings(write_config(tmp_path, ""))

        assert settings.personas.security == ["sensai.config.personas/security.toml"]

    def test_security_can_be_emptied(self, tmp_path: Path) -> None:
        assert load_settings(write_config(tmp_path, "[personas]\nsecurity = []\n")).personas.security == []

    def test_the_packaged_default_is_a_persona_file(self) -> None:
        package, _, filename = PersonaSettings().security[0].rpartition("/")

        assert resources.files(package).joinpath(filename).is_file()


class TestLogging:
    def test_is_read_from_logging_table(self, tmp_path: Path) -> None:
        settings = load_settings(write_config(tmp_path, "[logging]\nenabled = false\nlog_content = false\n"))

        assert settings.logging == LoggingSettings(enabled=False, log_content=False)


class TestContext:
    def test_is_read_from_context_table(self, tmp_path: Path) -> None:
        settings = load_settings(
            write_config(tmp_path, "[context]\nbudget = 4096\nwarning_threshold = 0.5\nkeep_recent = 2\n")
        )

        assert settings.context == ContextSettings(budget=4096, warning_threshold=0.5, keep_recent=2)

    def test_missing_table_leaves_the_budget_to_the_model(self, tmp_path: Path) -> None:
        assert load_settings(write_config(tmp_path, "")).context.budget is None


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
            ('[db]\nrag.location = ""\n', "db.rag.location must be a non-empty string"),
            ('[db]\nrag.url = "x"\n', "unexpected keyword argument 'url'"),
            ('[db]\nrag = "x"\n', "rag must be a table"),
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
            ('web = "x"\n', "web must be a table"),
            ("[web]\napi_key = 1\n", "web.api_key must be a string"),
            ('[web]\napi_base_url = "ollama.com"\n', "web.api_base_url must be an http\\(s\\) URL"),
            ("[web]\ntimeout = 0\n", "web.timeout must be a positive number"),
            ('[web]\ntimeout = "10"\n', "web.timeout must be a positive number"),
            ("[web]\ntimeout = true\n", "web.timeout must be a positive number"),
            ('[features]\ndisabled = "tools.x"\n', "features.disabled must be a list"),
            ('[features]\nconfirm = "tools.x"\n', "features.confirm must be a list"),
            ('[features]\nconfirm = [""]\n', "features.confirm entry must be a non-empty string"),
            ("[features]\nconfirm = [1]\n", "features.confirm entry must be a non-empty string"),
            ('personas = "x"\n', "personas must be a table"),
            ('[personas]\nsecurity = "a.toml"\n', "personas.security must be a list"),
            ('[personas]\nsecurity = [""]\n', "personas.security entry must be a non-empty string"),
            ("[personas]\nsecurity = [1]\n", "personas.security entry must be a non-empty string"),
            ('logging = "x"\n', "logging must be a table"),
            ('[logging]\nenabled = "yes"\n', "logging.enabled must be a boolean"),
            ("[logging]\nlog_content = 1\n", "logging.log_content must be a boolean"),
            ('context = "x"\n', "context must be a table"),
            ("[context]\nbudget = 0\n", "context.budget must be a positive number of tokens"),
            ("[context]\nbudget = 1.5\n", "context.budget must be a positive number of tokens"),
            ("[context]\nbudget = true\n", "context.budget must be a positive number of tokens"),
            ('[context]\nwarning_threshold = "x"\n', "context.warning_threshold must be a number"),
            ("[context]\nwarning_threshold = 0\n", "context.warning_threshold must be in"),
            ("[context]\nwarning_threshold = 1.5\n", "context.warning_threshold must be in"),
            ("[context]\nkeep_recent = 0\n", "context.keep_recent must be a positive number of messages"),
            ('[context]\nkeep_recent = "6"\n', "context.keep_recent must be a positive number of messages"),
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

    def test_redis_url_is_hidden_from_repr(self) -> None:
        assert "secret" not in repr(CacheSettings(redis_url="rediss://user:secret@cache.example.com"))

    def test_invalid_redis_url_is_not_echoed(self) -> None:
        with pytest.raises(ValueError, match=r"cache\.redis_url must be a redis://") as error:
            CacheSettings(redis_url="http://user:secret@cache.example.com")

        assert "secret" not in str(error.value)


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
