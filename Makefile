NAME := sensai

all: install

install:
	uv sync --locked
	uv run pre-commit install

run:
	uv run $(NAME)

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
