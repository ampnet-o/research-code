.PHONY: dbg test dev_install note_srv dev_dep_install setup

NOTE_HOST = 0.0.0.0

dbg:
	uv run python3 test/dbg.py

test:
	uv run pytest

note_srv:
	marimo edit --headless --host=$(NOTE_HOST)

dev_install:
	pip install -e .[gpu]

dev_dep_install:
	pip install --group dev

setup:
	uv sync --group gpu --group dev --group test

