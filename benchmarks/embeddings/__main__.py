"""Benchmark of the embedding models: `run <model>` measures one model, `report` renders the tables of every model run.

Runs against a real Ollama, outside of the tests: `make bench-embeddings MODEL=<model>`, then
`make bench-embeddings-report`.
"""

from __future__ import annotations

import argparse
import json
import sys
import tomllib
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from benchmarks import ollama
from benchmarks.embeddings.report import render
from benchmarks.embeddings.runner import EmbeddingBenchmark, separation
from sensai.config.settings import load_settings
from sensai.error import SensaiError
from sensai.llm.embedder import Embedder
from sensai.llm.provider_manager import ProviderManager

if TYPE_CHECKING:
    from collections.abc import Sequence

HERE = Path(__file__).parent
CASES = HERE / "cases.toml"
RESULTS = HERE / "results"


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command line."""
    parser = argparse.ArgumentParser(prog="python -m benchmarks.embeddings", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="benchmark one embedding model and save its results")
    run.add_argument("model", help="model to benchmark, as Ollama lists it, such as nomic-embed-text:latest")
    commands.add_parser("report", help="render the tables of every saved result")
    args = parser.parse_args(argv)

    if args.command == "report":
        return report()
    try:
        return benchmark(args.model)
    except SensaiError as err:
        print(f"error: {err}", file=sys.stderr)
        return 1


def benchmark(model: str) -> int:
    """Run the benchmark on the embedding `model`, and save the results next to the previous ones."""
    settings = load_settings()
    provider = ProviderManager(settings.llm).get_provider(model)
    if provider is None:
        print(f"error: model '{model}' is not available: ollama pull {model}", file=sys.stderr)
        return 1
    with CASES.open("rb") as file:
        cases = tomllib.load(file)
    bench = EmbeddingBenchmark(
        Embedder(provider, model), cases, progress=lambda line: print(line, file=sys.stderr, flush=True)
    )
    result: dict[str, Any] = {
        "model": model,
        "size_gb": ollama.size_gb(settings, model),
        "date": datetime.now(tz=UTC).isoformat(timespec="seconds"),
        "cost": bench.cost(provider),
        "retrieval": bench.retrieval(),
        "cache": bench.cache(),
    }
    result["separation"] = separation(result["cache"], settings.cache.similarity_threshold)
    result["gpu_share"] = ollama.gpu_share(settings, model)
    RESULTS.mkdir(exist_ok=True)
    path = RESULTS / f"{model.replace(':', '_').replace('/', '_')}.json"
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"results saved to {path}", file=sys.stderr)
    return 0


def report() -> int:
    """Render the tables of every saved result, smallest model first, into the README of the results folder."""
    results = [json.loads(path.read_text(encoding="utf-8")) for path in sorted(RESULTS.glob("*.json"))]
    if not results:
        print("error: no results yet, run `make bench-embeddings MODEL=<model>` first", file=sys.stderr)
        return 1
    results.sort(key=lambda result: result.get("size_gb") or 0)
    text = render(results)
    (RESULTS / "README.md").write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
