from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from importlib import resources
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path


class ConfigError(Exception):
    """Raised when the settings cannot be loaded or are invalid."""


@dataclass
class OllamaSettings:
    """Connection settings of the Ollama HTTP API."""

    base_url: str = "http://localhost:11434/"
    embedding_model: str = "nomic-embed-text"


@dataclass
class LLMSettings:
    """Model settings and the backend connection."""

    model: str = "qwen2.5:1.5b"
    ollama: OllamaSettings = field(default_factory=OllamaSettings)


@dataclass
class Settings:
    """Every runtime setting, built once by the composition root and injected."""

    llm: LLMSettings = field(default_factory=LLMSettings)


def load_settings(path: Path | None = None) -> Settings:
    """Load the settings file, falling back to the built-in defaults for missing keys.

    Without a path, the settings.toml packaged with sensai is used.
    """
    source = path if path is not None else resources.files("sensai.config").joinpath("settings.toml")
    try:
        with source.open("rb") as f:
            data = tomllib.load(f)
        llm = _pop_table(data, "llm")
        ollama = OllamaSettings(**_pop_table(llm, "ollama"))
        return Settings(llm=LLMSettings(**llm, ollama=ollama), **data)
    except FileNotFoundError:
        msg = f"config file not found: {source}"
    except IsADirectoryError:
        msg = f"config path is a directory: {source}"
    except tomllib.TOMLDecodeError as err:
        msg = f"invalid TOML in {source}: {err}"
    except TypeError as err:
        msg = f"invalid settings in {source}: {err}"
    raise ConfigError(msg)


def _pop_table(parent: dict[str, Any], key: str) -> dict[str, Any]:
    table = parent.pop(key, {})
    if not isinstance(table, dict):
        msg = f"{key} must be a table"
        raise TypeError(msg)
    return table
