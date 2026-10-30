NAME := sensai
MODEL = qwen3:1.7b
EMBEDDING_MODEL = nomic-embed-text:latest

all: install

SETTINGS := src/sensai/config/settings.toml

install: $(SETTINGS)
	uv sync --locked --all-packages
	uv run pre-commit install

$(SETTINGS):
	cp $(SETTINGS).template $(SETTINGS)

run: qdrant
	uv run $(NAME) --model ${MODEL}

run-server:
	uv run --package sensai-server sensai-server

run-tui:
	uv run --package sensai-tui sensai-tui --model ${MODEL}
bench:
	uv run python -m benchmarks.models run ${MODEL}

bench-report:
	uv run python -m benchmarks.models report

bench-embeddings:
	uv run python -m benchmarks.embeddings run ${EMBEDDING_MODEL}

bench-embeddings-report:
	uv run python -m benchmarks.embeddings report

qdrant:
	docker compose up -d --wait

qdrant-stop:
	docker compose down

help:
	uv run $(NAME) --help

tests:
	uv run pytest

lint:
	uv run pre-commit run --all-files

format:
	uv run ruff format
	uv run ruff check --fix

lock:
	uv lock

.PHONY: all install run run-server run-tui bench bench-report bench-embeddings bench-embeddings-report qdrant qdrant-stop tests lint format lock
