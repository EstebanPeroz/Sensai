from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from importlib import resources
from typing import TYPE_CHECKING, Any, ClassVar
from urllib.parse import urlsplit

from sensai.error import SensaiError

if TYPE_CHECKING:
    from importlib.resources.abc import Traversable
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
class WebSettings:
    """Settings of the [web] table, used by the web_search and web_fetch tools.

    `api_key` is a secret, set in the git-ignored settings.toml only. Without it, the keyless backends are used.
    """

    api_base_url: str = "https://ollama.com/"
    api_key: str = field(default="", repr=False)
    timeout: float = 10.0

    def __post_init__(self) -> None:
        """Check the values, which come unchecked from the settings file."""
        _check_name(self.api_base_url, "web.api_base_url")
        _check_url(self.api_base_url, "web.api_base_url")
        if not isinstance(self.api_key, str):
            msg = f"web.api_key must be a string, got {type(self.api_key).__name__}"
            raise TypeError(msg)
        if isinstance(self.timeout, bool) or not isinstance(self.timeout, int | float) or self.timeout <= 0:
            msg = f"web.timeout must be a positive number, got {self.timeout!r}"
            raise TypeError(msg)


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
class FeatureSettings:
    """Settings of the [features] table: the features to switch off; every other one is on.

    A feature is named `<namespace>.<name>`, such as `tools.rag_retrieval`.
    """

    disabled: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Check the disabled list, which comes unchecked from the settings file."""
        if not isinstance(self.disabled, list):
            msg = "features.disabled must be a list of feature names"
            raise TypeError(msg)
        for name in self.disabled:
            _check_name(name, "features.disabled entry")

    def disabled_in(self, namespace: str) -> set[str]:
        """Return the names, stripped of their namespace, of the features disabled in `namespace`."""
        prefix = f"{namespace}."
        return {name.removeprefix(prefix) for name in self.disabled if name.startswith(prefix)}


@dataclass
class LoggingSettings:
    """Settings of the [logging] table: the structured log of every call to the model."""

    enabled: bool = True
    log_content: bool = True

    def __post_init__(self) -> None:
        """Check the flags, which come unchecked from the settings file."""
        _check_bool(self.enabled, "logging.enabled")
        _check_bool(self.log_content, "logging.log_content")


@dataclass
class Settings:
    """Every runtime setting, built once by the composition root and injected."""

    llm: LLMSettings = field(default_factory=LLMSettings)
    web: WebSettings = field(default_factory=WebSettings)
    db: DBSettings = field(default_factory=DBSettings)
    features: FeatureSettings = field(default_factory=FeatureSettings)
    logging: LoggingSettings = field(default_factory=LoggingSettings)


def default_settings_file() -> Traversable:
    """Return the local settings.toml of sensai.config, or the settings.toml.template it is copied from."""
    config = resources.files("sensai.config")
    local = config.joinpath("settings.toml")
    return local if local.is_file() else config.joinpath("settings.toml.template")


def load_settings(path: Path | None = None) -> Settings:
    """Load the settings file, falling back to the built-in defaults for missing keys.

    Without a path, the local settings.toml is used, or its template when there is none.
    """
    source = path
    try:
        if source is None:
            source = default_settings_file()
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
            web = WebSettings(**_pop_table(data, "web"))
            db = _pop_table(data, "db")
            persistent = PersistentDBSettings(**_pop_table(db, "persistent"))
            features = FeatureSettings(**_pop_table(data, "features"))
            logging = LoggingSettings(**_pop_table(data, "logging"))
            return Settings(
                llm=LLMSettings(**llm, providers=providers),
                web=web,
                db=DBSettings(**db, persistent=persistent),
                features=features,
                logging=logging,
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


def _check_bool(value: object, name: str) -> None:
    if not isinstance(value, bool):
        msg = f"{name} must be a boolean"
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
