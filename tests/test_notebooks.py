"""End-to-end: each example notebook runs top-to-bottom (the real integration test).

Slow — run with ``pytest --run-notebooks`` (or ``make test-notebooks`` / the CI 'notebooks' job).
Requires the sample data: ``python scripts/make_sample_data.py``.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = sorted((ROOT / "notebooks").glob("0*.py"))
SAMPLE = ROOT / "data" / "raw" / "customers.parquet"


@pytest.mark.notebook
@pytest.mark.parametrize("notebook", NOTEBOOKS, ids=lambda p: p.stem)
def test_notebook_runs(notebook: Path) -> None:
    if not SAMPLE.exists():
        pytest.skip("sample data missing — run `python scripts/make_sample_data.py` first")
    env = {**os.environ, "MPLBACKEND": "Agg", "PYTHONIOENCODING": "utf-8"}
    result = subprocess.run(
        [sys.executable, str(notebook)],
        capture_output=True,
        text=True,
        env=env,
        timeout=900,
        cwd=ROOT,
    )
    assert result.returncode == 0, result.stderr[-3000:]


def test_notebooks_discovered() -> None:
    # guard: an empty/renamed glob must fail loudly, not pass with zero notebooks run
    assert len(NOTEBOOKS) >= 5
