from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from importlib import resources
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path


class ConfigError(Exception):
    """Raised when the settings cannot be loaded or are invalid."""


@dataclass
class LLMSettings:
    """Backend-agnostic model settings."""

    model: str = "qwen2.5:1.5b"


@dataclass
class OllamaSettings:
    """Connection settings of the Ollama HTTP API."""

    base_url: str = "http://localhost:11434/"
    embedding_model: str = "nomic-embed-text"


@dataclass
class Settings:
    """Every runtime setting, built once by the composition root and injected."""

    llm: LLMSettings = field(default_factory=LLMSettings)
    ollama: OllamaSettings = field(default_factory=OllamaSettings)


def load_settings(path: Path | None = None) -> Settings:
    """Load the settings file, falling back to the built-in defaults for missing keys.

    Without a path, the settings.toml packaged with sensai is used.
    """
    source = path if path is not None else resources.files("sensai.config").joinpath("settings.toml")
    try:
        with source.open("rb") as f:
            data = tomllib.load(f)
        return Settings(
            llm=LLMSettings(**data.get("llm", {})),
            ollama=OllamaSettings(**data.get("ollama", {})),
        )
    except FileNotFoundError:
        msg = f"config file not found: {source}"
    except IsADirectoryError:
        msg = f"config path is a directory: {source}"
    except tomllib.TOMLDecodeError as err:
        msg = f"invalid TOML in {source}: {err}"
    except TypeError as err:
        msg = f"invalid settings in {source}: {err}"
    raise ConfigError(msg)
