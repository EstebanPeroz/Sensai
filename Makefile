MODEL = qwen3:1.7b

all: install

install:
	uv sync --locked --all-packages
	uv run pre-commit install

run-server: docker
	uv run --package sensai-server sensai-server --config ./packages/server/settings.toml

run-tui:
	uv run --package sensai-tui sensai-tui --model ${MODEL}

docker:
	docker compose up -d --wait

docker-stop:
	docker compose down

help:
	uv run --package sensai-tui sensai-tui --help
	uv run --package sensai-server sensai-server --help

tests:
	uv run pytest

lint:
	uv run pre-commit run --all-files

format:
	uv run ruff format
	uv run ruff check --fix

lock:
	uv lock

.PHONY: all install run-server run-tui docker docker-stop tests lint format lock
