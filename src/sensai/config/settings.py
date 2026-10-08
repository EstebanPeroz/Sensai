from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from importlib import resources
from typing import TYPE_CHECKING, Any, ClassVar
from urllib.parse import urlsplit

from sensai.error import SensaiError

if TYPE_CHECKING:
    from pathlib import Path


class ConfigError(SensaiError):
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
    providers: dict[str, ProviderSettings] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Check the embedding model, which comes unchecked from the settings file."""
        if self.embedding_model is not None:
            _check_name(self.embedding_model, "llm.embedding_model")


@dataclass
class PersistentDBSettings:
    """Settings of the [db.persistent] table: the persistent database."""

    url: str | None = None

    def __post_init__(self) -> None:
        """Check the URL, which comes unchecked from the settings file."""
        if self.url is not None:
            _check_name(self.url, "db.persistent.url")


@dataclass
class DBSettings:
    """Settings of the [db] table."""

    persistent: PersistentDBSettings = field(default_factory=PersistentDBSettings)


@dataclass
class Settings:
    """Every runtime setting, built once by the composition root and injected."""

    llm: LLMSettings = field(default_factory=LLMSettings)
    db: DBSettings = field(default_factory=DBSettings)


def load_settings(path: Path | None = None) -> Settings:
    """Load the settings file, falling back to the built-in defaults for missing keys.

    Without a path, the settings.toml packaged with sensai is used.
    """
    source = path
    try:
        if source is None:
            source = resources.files("sensai.config").joinpath("settings.toml")
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
            providers: dict[str, ProviderSettings] = {
                provider.name: provider(**_pop_table(llm, provider.name))
                for provider in PROVIDERS
                if provider.name in llm
            }
            db = _pop_table(data, "db")
            persistent = PersistentDBSettings(**_pop_table(db, "persistent"))
            return Settings(
                llm=LLMSettings(**llm, providers=providers),
                db=DBSettings(**db, persistent=persistent),
                **data,
            )
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
