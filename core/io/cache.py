"""``@cached`` — transparent Parquet caching of expensive loaders to ``data/interim``.

Wrap any ``-> pl.DataFrame`` loader; the result is cached by function + arguments. The wrapped
function gains a ``refresh`` keyword (``refresh=True`` recomputes and overwrites) — the modern
replacement for the old manual ``load_local_data`` / ``save_local_data`` toggle.
"""

from __future__ import annotations

import functools
import hashlib
from collections.abc import Callable

import polars as pl

from core.config import ROOT

CACHE_DIR = ROOT / "data" / "interim"


def cached(func: Callable[..., pl.DataFrame]) -> Callable[..., pl.DataFrame]:
    """Cache a ``-> pl.DataFrame`` loader to Parquet, keyed by function + arguments."""

    @functools.wraps(func)
    def wrapper(*args: object, refresh: bool = False, **kwargs: object) -> pl.DataFrame:
        kw = sorted(kwargs.items(), key=lambda item: item[0])
        key = f"{func.__module__}.{func.__qualname__}|{args!r}|{kw!r}"
        digest = hashlib.sha1(key.encode()).hexdigest()[:16]
        path = CACHE_DIR / f"{func.__name__}-{digest}.parquet"
        if path.is_file() and not refresh:
            return pl.read_parquet(path)
        result = func(*args, **kwargs)
        path.parent.mkdir(parents=True, exist_ok=True)
        result.write_parquet(path)
        return result

    return wrapper
