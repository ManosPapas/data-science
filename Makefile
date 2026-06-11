.PHONY: sync lint format type test check lab notebooks

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

lab:         ## launch JupyterLab in the browser
	uv run jupyter lab

notebooks:   ## sync jupytext .py <-> .ipynb pairs
	uv run jupytext --sync notebooks/*.py
