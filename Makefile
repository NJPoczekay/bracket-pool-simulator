.PHONY: bootstrap sync lint format typecheck test ci precommit-install

bootstrap: sync

sync:
	uv sync --extra dev

lint:
	uv run --extra dev ruff check .

format:
	uv run --extra dev ruff format .

typecheck:
	uv run --extra dev mypy src tests

test:
	uv run --extra dev pytest

ci: lint typecheck test

precommit-install:
	uv run --extra dev pre-commit install
