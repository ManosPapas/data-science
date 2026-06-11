"""Named, versioned model store — save/load fitted models so you can find the newest.

Models live under ``models/<name>/v<N>/`` (gitignored): ``model.joblib`` + a ``metadata.json``
sidecar (timestamp plus any metrics/params you pass). ``load_model`` returns the newest by default.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib

from core.config import ROOT

MODELS_DIR = ROOT / "models"


def model_versions(name: str) -> list[int]:
    """Sorted version numbers stored for ``name`` (empty if none)."""
    directory = MODELS_DIR / name
    if not directory.is_dir():
        return []
    return sorted(int(p.name[1:]) for p in directory.glob("v*") if p.name[1:].isdigit())


def save_model(
    model: Any, name: str, *, version: int | None = None, metadata: dict[str, Any] | None = None
) -> Path:
    """Save a fitted model under ``name``, auto-incrementing the version; returns the dir."""
    next_version = version if version is not None else max(model_versions(name), default=0) + 1
    target = MODELS_DIR / name / f"v{next_version}"
    target.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, target / "model.joblib")
    info = {
        "name": name,
        "version": next_version,
        "saved_at": datetime.now(UTC).isoformat(),
        **(metadata or {}),
    }
    (target / "metadata.json").write_text(json.dumps(info, indent=2, default=str), encoding="utf-8")
    return target


def load_model(name: str, *, version: int | None = None) -> Any:
    """Load a stored model — the newest version by default, or a specific ``version``."""
    chosen = version if version is not None else max(model_versions(name), default=0)
    path = MODELS_DIR / name / f"v{chosen}" / "model.joblib"
    if not path.is_file():
        raise FileNotFoundError(f"no model {name!r} version {chosen}")
    return joblib.load(path)


def list_models() -> list[str]:
    """Names of all stored models."""
    if not MODELS_DIR.is_dir():
        return []
    return sorted(p.name for p in MODELS_DIR.iterdir() if p.is_dir())
