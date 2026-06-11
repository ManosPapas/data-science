"""One-command setup for a fresh clone — no uv required.

Creates a local virtual environment, installs the package with all extras and dev tools, registers
the Jupyter kernel, and builds the sample datasets. Run with any Python 3.13+:

    python scripts/bootstrap.py

Everything goes through ``python -m`` so it works on locked-down machines where pip's ``.exe`` shims
are blocked. (If you have uv, ``uv sync --all-extras`` installs the packages instead.)
"""

from __future__ import annotations

import subprocess
import sys
import venv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VENV = ROOT / ".venv"
EXTRAS = "geo,explain,excel,gam,imbalance,interactive"
DEV_TOOLS = [
    "jupyterlab",
    "jupytext",
    "ipykernel",
    "ruff",
    "mypy",
    "pytest",
    "pre-commit",
    "nbstripout",
    "types-PyYAML",
]


def _venv_python() -> str:
    bin_dir = "Scripts" if sys.platform == "win32" else "bin"
    exe = "python.exe" if sys.platform == "win32" else "python"
    return str(VENV / bin_dir / exe)


def _run(args: list[str]) -> None:
    print(">", " ".join(args))
    subprocess.run(args, check=True, cwd=ROOT)


def main() -> None:
    # ruff assumes 3.13 (requires-python), but a fresh user may launch this with an older python:
    if sys.version_info < (3, 13):  # noqa: UP036
        sys.exit(f"Python 3.13+ required; this interpreter is {sys.version.split()[0]}")

    if not VENV.exists():
        print(f"Creating virtual environment at {VENV} ...")
        venv.create(VENV, with_pip=True)

    py = _venv_python()
    _run([py, "-m", "pip", "install", "--upgrade", "pip"])
    _run([py, "-m", "pip", "install", "-e", f".[{EXTRAS}]"])
    _run([py, "-m", "pip", "install", *DEV_TOOLS])
    _run(
        [
            py,
            "-m",
            "ipykernel",
            "install",
            "--user",
            "--name",
            "data-science",
            "--display-name",
            "Python (data-science)",
        ]
    )
    _run([py, "scripts/make_sample_data.py"])

    print("\nDone! Activate the venv and launch JupyterLab:")
    print(r"  .venv\Scripts\Activate.ps1   (bash/macOS: source .venv/bin/activate)")
    print("  python -m jupyterlab")


if __name__ == "__main__":
    main()
