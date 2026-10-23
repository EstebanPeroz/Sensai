from importlib import resources
from pathlib import Path

import pytest

from sensai.config.settings import (
    PROVIDERS,
    ConfigError,
    LLMSettings,
    LoggingSettings,
    OllamaSettings,
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

        assert settings.llm.embedding_model == "nomic-embed-text"
        assert settings.llm.providers == {"ollama": OllamaSettings()}
        assert settings.web == WebSettings()
        assert settings.logging == LoggingSettings(enabled=True, log_content=True)

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


class TestWeb:
    def test_is_read_from_web_table(self, tmp_path: Path) -> None:
        content = '[web]\napi_base_url = "https://example.com"\napi_key = "secret"\ntimeout = 3\n'
        settings = load_settings(write_config(tmp_path, content))

        assert settings.web == WebSettings(api_base_url="https://example.com", api_key="secret", timeout=3)

    def test_api_key_is_hidden_from_repr(self) -> None:
        assert "secret" not in repr(WebSettings(api_key="secret"))


class TestLogging:
    def test_is_read_from_logging_table(self, tmp_path: Path) -> None:
        settings = load_settings(write_config(tmp_path, "[logging]\nenabled = false\nlog_content = false\n"))

        assert settings.logging == LoggingSettings(enabled=False, log_content=False)


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
            ('web = "x"\n', "web must be a table"),
            ("[web]\napi_key = 1\n", "web.api_key must be a string"),
            ('[web]\napi_base_url = "ollama.com"\n', "web.api_base_url must be an http\\(s\\) URL"),
            ("[web]\ntimeout = 0\n", "web.timeout must be a positive number"),
            ('[web]\ntimeout = "10"\n', "web.timeout must be a positive number"),
            ("[web]\ntimeout = true\n", "web.timeout must be a positive number"),
            ('logging = "x"\n', "logging must be a table"),
            ('[logging]\nenabled = "yes"\n', "logging.enabled must be a boolean"),
            ("[logging]\nlog_content = 1\n", "logging.log_content must be a boolean"),
        ],
    )
    def test_invalid_settings_raise_config_error(self, tmp_path: Path, content: str, reason: str) -> None:
        with pytest.raises(ConfigError, match=f"invalid settings in .*{reason}"):
            load_settings(write_config(tmp_path, content))


class TestBaseUrl:
    @pytest.mark.parametrize("url", ["http://localhost:11434/", "https://ollama.example.com", "http://127.0.0.1"])
    def test_accepts_http_urls(self, url: str) -> None:
        assert OllamaSettings(base_url=url).base_url == url


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
