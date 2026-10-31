NAME := sensai
MODEL = qwen3:1.7b

all: install

SETTINGS := src/sensai/config/settings.toml

install: $(SETTINGS)
	uv sync --locked --all-packages
	uv run pre-commit install

$(SETTINGS):
	cp $(SETTINGS).template $(SETTINGS)

run: docker
	uv run $(NAME) --model ${MODEL}

run-server: docker
	uv run --package sensai-server sensai-server --config ./packages/server/settings.toml

run-tui:
	uv run --package sensai-tui sensai-tui --model ${MODEL}

docker:
	docker compose up -d --wait

docker-stop:
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

.PHONY: all install run run-server run-tui docker docker-stop tests lint format lock
