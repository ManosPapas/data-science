"""Typed application settings and layered, non-secret YAML config.

Secrets come from the environment / ``.env`` (typed, opaque ``SecretStr``); non-secret
configuration comes from ``conf/*.yaml``. Nothing else should read ``os.environ`` directly —
go through :func:`get_settings`.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[1]
CONF_DIR = ROOT / "conf"


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    with path.open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    return data if isinstance(data, dict) else {}


def _deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


class DbSecret(BaseModel):
    """Per-database secret, populated from ``<NAME>__PASSWORD`` in ``.env``."""

    password: SecretStr | None = None


class Settings(BaseSettings):
    """Typed settings. Add a ``DbSecret`` field per database — its name must match the
    key in ``conf/databases.yaml`` and the ``<NAME>__`` env prefix."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        extra="ignore",
    )

    environment: str = "dev"

    # Databases — one field per connection (name ties together conf/databases.yaml,
    # the .env prefix WAREHOUSE__PASSWORD, and a future get_engine("warehouse")).
    warehouse: DbSecret = DbSecret()

    # External API
    api_token: SecretStr | None = None

    @property
    def conf(self) -> dict[str, Any]:
        """Non-secret YAML config, with per-environment overrides deep-merged in."""
        base = _read_yaml(CONF_DIR / "config.yaml")
        overlay = _read_yaml(CONF_DIR / f"config.{self.environment}.yaml")
        merged = _deep_merge(base, overlay)
        merged["databases"] = _read_yaml(CONF_DIR / "databases.yaml")
        return merged


@lru_cache
def get_settings() -> Settings:
    """Return the cached settings singleton."""
    return Settings()
