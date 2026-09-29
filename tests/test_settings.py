from pathlib import Path

import pytest

from sensai.config.settings import PROVIDERS, ConfigError, LLMSettings, OllamaSettings, Settings, load_settings


def write_config(tmp_path: Path, content: str) -> Path:
    path = tmp_path / "settings.toml"
    path.write_text(content)
    return path


class TestPackagedSettings:
    def test_loads_packaged_file_by_default(self) -> None:
        settings = load_settings()

        assert isinstance(settings, Settings)
        assert settings.llm.model in settings.llm.models
        assert settings.llm.providers == [OllamaSettings()]


class TestPrecedence:
    def test_missing_keys_fall_back_to_defaults(self, tmp_path: Path) -> None:
        settings = load_settings(write_config(tmp_path, '[llm.ollama]\nbase_url = "http://ollama:1234"\n'))

        assert settings.llm.providers == [OllamaSettings(base_url="http://ollama:1234")]
        assert settings.llm.embedding_model == LLMSettings().embedding_model
        assert settings.llm.model == LLMSettings().model

    def test_empty_file_gives_defaults(self, tmp_path: Path) -> None:
        assert load_settings(write_config(tmp_path, "")) == Settings()


class TestProviders:
    def test_no_provider_table_gives_no_provider(self, tmp_path: Path) -> None:
        assert load_settings(write_config(tmp_path, '[llm]\nembedding_model = "x"\n')).llm.providers == []

    def test_empty_provider_table_uses_provider_defaults(self, tmp_path: Path) -> None:
        assert load_settings(write_config(tmp_path, "[llm.ollama]\n")).llm.providers == [OllamaSettings()]

    def test_every_supported_provider_has_a_unique_name(self) -> None:
        names = [provider.name for provider in PROVIDERS]

        assert len(names) == len(set(names))


class TestModels:
    def test_first_model_is_active(self, tmp_path: Path) -> None:
        settings = load_settings(write_config(tmp_path, '[llm]\nmodels = ["llama3.2", "mistral"]\n'))

        assert settings.llm.model == "llama3.2"
        assert settings.llm.models == ["llama3.2", "mistral"]


class TestEmbeddingModel:
    def test_is_read_from_llm_table(self, tmp_path: Path) -> None:
        settings = load_settings(write_config(tmp_path, '[llm]\nembedding_model = "embeddinggemma"\n'))

        assert settings.llm.embedding_model == "embeddinggemma"

    def test_packaged_file_sets_it(self) -> None:
        assert load_settings().llm.embedding_model is not None


class TestSelectModel:
    def test_selects_supported_model(self) -> None:
        llm = LLMSettings(models=["llama3.2", "mistral"])

        llm.select_model("mistral")

        assert llm.model == "mistral"

    def test_rejects_unsupported_model(self) -> None:
        llm = LLMSettings(models=["llama3.2", "mistral"])

        with pytest.raises(ConfigError, match=r"unknown model 'qwen3', available models: llama3\.2, mistral"):
            llm.select_model("qwen3")
        assert llm.model == "llama3.2"


class TestValidation:
    @pytest.mark.parametrize(
        ("content", "reason"),
        [
            ('[llm]\nmodle = "x"\n', "unexpected keyword argument 'modle'"),
            ('[ollama]\nbase_url = "x"\n', "unexpected keyword argument 'ollama'"),
            ('[llm.ollama]\nurl = "x"\n', "unexpected keyword argument 'url'"),
            ('[llm.mistral]\napi_key = "x"\n', "unexpected keyword argument 'mistral'"),
            ('[llm]\nmodel = "llama3.2"\n', "unexpected keyword argument 'model'"),
            ('llm = "x"\n', "llm must be a table"),
            ('[llm]\nollama = "x"\n', "ollama must be a table"),
            ('[llm]\nmodels = "llama3.2"\n', "llm.models must be a non-empty list"),
            ("[llm]\nmodels = []\n", "llm.models must be a non-empty list"),
            ('[llm]\nmodels = ["llama3.2", 1]\n', "each llm.models entry must be a non-empty string"),
            ('[llm]\nmodels = ["llama3.2", ""]\n', "each llm.models entry must be a non-empty string"),
            ('[llm.ollama]\nbase_url = ["a"]\n', "llm.ollama.base_url must be a non-empty string"),
            ('[llm.ollama]\nembedding_model = "x"\n', "unexpected keyword argument 'embedding_model'"),
            ('[llm]\nembedding_model = ""\n', "llm.embedding_model must be a non-empty string"),
            ('[llm.ollama]\nbase_url = "http://"\n', "llm.ollama.base_url must be an http\\(s\\) URL with a host"),
            ('[llm.ollama]\nbase_url = "localhost:11434"\n', "llm.ollama.base_url must be an http\\(s\\) URL"),
            ('[llm.ollama]\nbase_url = "ftp://ollama:11434"\n', "llm.ollama.base_url must be an http\\(s\\) URL"),
            ('[llm.ollama]\nbase_url = "http://ollama:port"\n', "llm.ollama.base_url must be an http\\(s\\) URL"),
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
            load_settings(write_config(tmp_path, "[llm\nmodel = \n"))
