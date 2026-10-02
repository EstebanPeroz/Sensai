from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from importlib import resources
from typing import TYPE_CHECKING, Any, ClassVar
from urllib.parse import urlsplit

if TYPE_CHECKING:
    from pathlib import Path


class ConfigError(Exception):
    """Raised when the settings cannot be loaded or are invalid."""


@dataclass
class ProviderSettings:
    """Settings of one LLM provider, read from its [llm.<name>] table."""

    name: ClassVar[str]

    base_url: str

    def __post_init__(self) -> None:
        """Check the base URL, which comes unchecked from the settings file."""
        _check_name(self.base_url, f"llm.{self.name}.base_url")
        _check_url(self.base_url, f"llm.{self.name}.base_url")


@dataclass
class OllamaSettings(ProviderSettings):
    """Connection settings of the Ollama HTTP API."""

    name: ClassVar[str] = "ollama"

    base_url: str = "http://localhost:11434/"


# Every supported provider: its settings are built when its [llm.<name>] table is in the settings file.
PROVIDERS: tuple[type[ProviderSettings], ...] = (OllamaSettings,)


@dataclass
class LLMSettings:
    """Settings of the [llm] table: the embedding model, shared by every provider, and the configured providers."""

    embedding_model: str | None = None
    providers: list[ProviderSettings] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Check the embedding model, which comes unchecked from the settings file."""
        if self.embedding_model is not None:
            _check_name(self.embedding_model, "llm.embedding_model")


@dataclass
class CacheSettings:
    """Settings of the [cache] table: the Redis server backing the semantic cache and when an answer is reused."""

    redis_url: str = "redis://localhost:6379/0"
    similarity_threshold: float = 0.92
    ttl: int = 86400

    def __post_init__(self) -> None:
        """Check the values, which come unchecked from the settings file."""
        _check_name(self.redis_url, "cache.redis_url")
        _check_redis_url(self.redis_url, "cache.redis_url")
        if isinstance(self.similarity_threshold, bool) or not isinstance(self.similarity_threshold, (int, float)):
            msg = "cache.similarity_threshold must be a number"
            raise TypeError(msg)
        if not 0 < self.similarity_threshold <= 1:
            msg = f"cache.similarity_threshold must be in ]0, 1], got {self.similarity_threshold}"
            raise ValueError(msg)
        if isinstance(self.ttl, bool) or not isinstance(self.ttl, int):
            msg = "cache.ttl must be an integer"
            raise TypeError(msg)
        if self.ttl <= 0:
            msg = f"cache.ttl must be a positive number of seconds, got {self.ttl}"
            raise ValueError(msg)


@dataclass
class Settings:
    """Every runtime setting, built once by the composition root and injected."""

    llm: LLMSettings = field(default_factory=LLMSettings)
    cache: CacheSettings = field(default_factory=CacheSettings)


def load_settings(path: Path | None = None) -> Settings:
    """Load the settings file, falling back to the built-in defaults for missing keys.

    Without a path, the settings.toml packaged with sensai is used.
    """
    source = path if path is not None else resources.files("sensai.config").joinpath("settings.toml")
    try:
        with source.open("rb") as f:
            data = tomllib.load(f)
    except FileNotFoundError:
        msg = f"config file not found: {source}"
    except IsADirectoryError:
        msg = f"config path is a directory: {source}"
    except OSError as err:
        msg = f"cannot read config file {source}: {err.strerror}"
    except UnicodeDecodeError:
        msg = f"config file is not valid UTF-8: {source}"
    except tomllib.TOMLDecodeError as err:
        msg = f"invalid TOML in {source}: {err}"
    else:
        try:
            llm = _pop_table(data, "llm")
            providers = [provider(**_pop_table(llm, provider.name)) for provider in PROVIDERS if provider.name in llm]
            cache = CacheSettings(**_pop_table(data, "cache"))
            return Settings(llm=LLMSettings(**llm, providers=providers), cache=cache, **data)
        except (TypeError, ValueError) as err:
            msg = f"invalid settings in {source}: {err}"
    raise ConfigError(msg)


def _pop_table(parent: dict[str, Any], key: str) -> dict[str, Any]:
    table = parent.pop(key, {})
    if not isinstance(table, dict):
        msg = f"{key} must be a table"
        raise TypeError(msg)
    return table


def _check_name(value: object, name: str) -> None:
    if not isinstance(value, str) or not value:
        msg = f"{name} must be a non-empty string"
        raise TypeError(msg)


def _check_url(value: str, name: str) -> None:
    msg = f"{name} must be an http(s) URL with a host, got {value!r}"
    parts = urlsplit(value)
    try:
        _ = parts.port  # raises ValueError on an invalid port
    except ValueError:
        raise ValueError(msg) from None
    if parts.scheme not in {"http", "https"} or not parts.hostname:
        raise ValueError(msg)


def _check_redis_url(value: str, name: str) -> None:
    msg = f"{name} must be a redis://, rediss:// or unix:// URL, got {value!r}"
    parts = urlsplit(value)
    try:
        _ = parts.port
    except ValueError:
        raise ValueError(msg) from None
    if parts.scheme == "unix":
        if not parts.path:
            raise ValueError(msg)
    elif parts.scheme not in {"redis", "rediss"} or not parts.hostname:
        raise ValueError(msg)
