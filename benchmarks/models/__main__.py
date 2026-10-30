"""Benchmark of the chat models: `run <model>` measures one model, `report` renders the tables of every model run.

Runs against a real Ollama, outside of the tests: `make bench MODEL=<model>`, then `make bench-report`.
"""

from __future__ import annotations

import argparse
import json
import sys
import tomllib
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import requests

from benchmarks.models.report import render
from benchmarks.models.runner import Benchmark
from sensai.config.settings import load_settings
from sensai.error import SensaiError
from sensai.llm.embedder import Embedder
from sensai.llm.provider_manager import ProviderManager
from sensai.tools.ask_user import AskUserTool
from sensai.tools.web.backend.fetch import FetchChain
from sensai.tools.web.backend.search import SearchChain
from sensai.tools.web.fetch import WebFetchTool
from sensai.tools.web.search import WebSearchTool

if TYPE_CHECKING:
    from collections.abc import Sequence

    from sensai.config.settings import Settings
    from sensai.tools.tool import Tool

HERE = Path(__file__).parent
CASES = HERE / "cases.toml"
RESULTS = HERE / "results"
CATEGORIES = ("latency", "quality", "tools", "instructions", "memory")
OLLAMA_TIMEOUT = 5


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command line."""
    parser = argparse.ArgumentParser(prog="python -m benchmarks.models", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="benchmark one model and save its results")
    run.add_argument("model", help="model to benchmark, as Ollama lists it, such as qwen3:1.7b")
    run.add_argument("--only", nargs="+", choices=CATEGORIES, default=CATEGORIES, help="categories to run")
    commands.add_parser("report", help="render the tables of every saved result")
    args = parser.parse_args(argv)

    if args.command == "report":
        return report()
    try:
        return benchmark(args.model, args.only)
    except SensaiError as err:
        print(f"error: {err}", file=sys.stderr)
        return 1


def benchmark(model: str, categories: Sequence[str]) -> int:
    """Run the `categories` of the benchmark on `model`, and save the results next to the previous ones."""
    settings = load_settings()
    provider_manager = ProviderManager(settings.llm)
    if not provider_manager.set_model(model):
        print(f"error: model '{model}' is not available: ollama pull {model}", file=sys.stderr)
        return 1
    with CASES.open("rb") as file:
        cases = tomllib.load(file)
    bench = Benchmark(
        provider_manager,
        _tools(settings, provider_manager),
        cases,
        context_window=settings.context.budget,
        progress=lambda line: print(line, file=sys.stderr, flush=True),
    )

    path = RESULTS / f"{model.replace(':', '_').replace('/', '_')}.json"
    result: dict[str, Any] = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    result |= {
        "model": model,
        "size_gb": _size_gb(settings, model),
        "date": datetime.now(tz=UTC).isoformat(timespec="seconds"),
        "context_window": settings.context.budget,
    }
    for category in categories:
        result[category] = getattr(bench, category)()
    # Read while the model is still loaded: a model too large for the GPU runs partly on the CPU, much slower.
    result["gpu_share"] = _gpu_share(settings, model)
    RESULTS.mkdir(exist_ok=True)
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"results saved to {path}", file=sys.stderr)
    return 0


def report() -> int:
    """Render the tables of every saved result, smallest model first, into the README of the results folder."""
    results = [json.loads(path.read_text(encoding="utf-8")) for path in sorted(RESULTS.glob("*.json"))]
    if not results:
        print("error: no results yet, run `make bench MODEL=<model>` first", file=sys.stderr)
        return 1
    results.sort(key=lambda result: result.get("size_gb") or 0)
    text = render(results)
    (RESULTS / "README.md").write_text(text, encoding="utf-8")
    print(text)
    return 0


def _tools(settings: Settings, provider_manager: ProviderManager) -> list[Tool]:
    """Build the tools the chat offers, only for what the model reads of them: the benchmark scripts their results.

    Their backends are left empty, since they never run.
    """
    embedder = Embedder(provider_manager.provider(), settings.llm.embedding_model or "")
    return [WebSearchTool(SearchChain([])), WebFetchTool(FetchChain([]), embedder), AskUserTool(lambda _question: None)]


def _size_gb(settings: Settings, model: str) -> float | None:
    """Size of `model` on disk, as Ollama lists it, None when it cannot tell."""
    sizes = [entry["size"] for entry in _ollama_models(settings, "api/tags") if entry.get("name") == model]
    return round(sizes[0] / 1e9, 2) if sizes else None


def _gpu_share(settings: Settings, model: str) -> float | None:
    """Share of the loaded `model` that sits in the memory of the GPU, None when it is not loaded."""
    loaded = [entry for entry in _ollama_models(settings, "api/ps") if entry.get("name") == model]
    if not loaded or not loaded[0].get("size"):
        return None
    return round(loaded[0].get("size_vram", 0) / loaded[0]["size"], 2)


def _ollama_models(settings: Settings, endpoint: str) -> list[dict[str, Any]]:
    """Models listed by the Ollama `endpoint`, `api/tags` for the installed ones or `api/ps` for the loaded ones."""
    provider = settings.llm.providers.get("ollama")
    if provider is None:
        return []
    try:
        response = requests.get(f"{provider.base_url.rstrip('/')}/{endpoint}", timeout=OLLAMA_TIMEOUT)
        return response.json().get("models", [])
    except requests.RequestException, ValueError:
        return []


if __name__ == "__main__":
    sys.exit(main())
