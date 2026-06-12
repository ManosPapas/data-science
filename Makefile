.PHONY: sync lint format type test check check-local test-notebooks pipeline lab notebooks

sync:        ## create .venv and install from the lockfile
	uv sync

lint:        ## ruff lint
	uv run ruff check .

format:      ## ruff format
	uv run ruff format .

type:        ## mypy (strict) on the package
	uv run mypy core

test:        ## run the test suite
	uv run pytest -q

check: lint type test  ## lint + types + tests

check-local: ## lint + types + tests without uv (python -m; for locked-down machines)
	python -m ruff check .
	python -m mypy core
	python -m pytest -q

test-notebooks: ## end-to-end: run every example notebook (builds sample data first)
	uv sync --all-extras
	uv run python scripts/make_sample_data.py
	uv run pytest --run-notebooks -m notebook -q

pipeline: ## full local gate — lint, format, types, unit tests + coverage floor, all notebooks
	python -m ruff check .
	python -m ruff format --check .
	python -m mypy core
	python -m pytest -q --cov=core --cov-report=term-missing:skip-covered --cov-fail-under=55
	python scripts/make_sample_data.py
	python -m pytest --run-notebooks -m notebook -q

lab:         ## launch JupyterLab in the browser
	uv run jupyter lab

notebooks:   ## sync jupytext .py <-> .ipynb pairs
	uv run jupytext --sync notebooks/*.py
