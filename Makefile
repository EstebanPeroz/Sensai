NAME := sensai
MODEL = qwen3:1.7b

all: install

install:
	uv sync --locked
	uv run pre-commit install

run:
	uv run $(NAME) --model ${MODEL}

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

.PHONY: all install run tests lint format lock
