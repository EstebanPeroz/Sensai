NAME := sensai
MODEL = qwen3:1.7b

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

.PHONY: all install run run-server run-tui qdrant qdrant-stop tests lint format lock
