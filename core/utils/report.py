"""Assemble metrics, tables, and chart images into a simple self-contained HTML report."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def html_report(
    title: str,
    *,
    metrics: Mapping[str, Any] | None = None,
    frames: Mapping[str, Any] | None = None,
    charts: Sequence[str | Path] | None = None,
    save: str | Path | None = None,
) -> str:
    """Build (and optionally save) an HTML report from metrics, frames, and chart image paths."""
    parts = [f"<h1>{title}</h1>", f"<p>Generated {datetime.now(UTC):%Y-%m-%d %H:%M UTC}</p>"]
    if metrics:
        rows = "".join(
            f"<tr><td>{name}</td><td>{value}</td></tr>" for name, value in metrics.items()
        )
        parts.append(f"<h2>Metrics</h2><table border='1'>{rows}</table>")
    if frames:
        for name, frame in frames.items():
            parts.append(f"<h2>{name}</h2>{frame.to_pandas().to_html(index=False)}")
    if charts:
        parts.extend(f'<img src="{path}" style="max-width:100%">' for path in charts)
    html = "<!doctype html><html><body>" + "".join(parts) + "</body></html>"
    if save is not None:
        Path(save).write_text(html, encoding="utf-8")
    return html
