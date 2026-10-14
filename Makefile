# Local equivalents of the CI jobs. `make check` runs all three.
.PHONY: install check lint types test test-fast

# Editable install next to editable joint-align and warpalign checkouts (--no-deps keeps pip from
# replacing them with the commits pinned in pyproject.toml). Fresh environments: pip install -e ".[dev]".
install:
	python3.11 -m pip install --no-deps -e .

check: lint types test

lint:
	ruff check .

types:
	mypy

test:
	python3.11 -m coverage run -m pytest && python3.11 -m coverage report

test-fast:
	python3.11 -m pytest -m "not slow"
