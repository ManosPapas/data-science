"""Typed settings + named database/API connections.

Non-secret config and connection *templates* live in committed ``conf/*.yaml``. Real secrets
(passwords, tokens) go in gitignored ``conf/*.local.yaml`` of the same shape and are deep-merged on
top — so nothing secret is ever committed. App-level environment still comes from ``.env``.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[1]
CONF_DIR = ROOT / "conf"


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    with path.open(encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    return loaded if isinstance(loaded, dict) else {}


def _deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _merged(name: str) -> dict[str, Any]:
    """``conf/<name>.yaml`` with gitignored ``conf/<name>.local.yaml`` deep-merged on top."""
    return _deep_merge(
        _read_yaml(CONF_DIR / f"{name}.yaml"), _read_yaml(CONF_DIR / f"{name}.local.yaml")
    )


class Settings(BaseSettings):
    """App-level settings from ``.env`` (connection secrets live in conf/*.local.yaml)."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    environment: str = "dev"

    @property
    def conf(self) -> dict[str, Any]:
        """Non-secret YAML config, with per-environment overrides deep-merged in."""
        base = _read_yaml(CONF_DIR / "config.yaml")
        overlay = _read_yaml(CONF_DIR / f"config.{self.environment}.yaml")
        return _deep_merge(base, overlay)


@lru_cache
def get_settings() -> Settings:
    """Return the cached settings singleton."""
    return Settings()


class ConnectionConfig(BaseModel):
    """A database connection — the classic ODBC block; YAML keys are case-insensitive."""

    model_config = ConfigDict(extra="ignore")

    server: str = ""
    user_name: str = ""
    password: SecretStr | None = None
    database: str = ""
    driver: str = "ODBC Driver 18 for SQL Server"


class ApiConfig(BaseModel):
    """A named HTTP API: base URL, non-secret headers, and an optional bearer token."""

    model_config = ConfigDict(extra="ignore")

    base_url: str = ""
    token: SecretStr | None = None
    headers: dict[str, str] = Field(default_factory=dict)


def get_connection(name: str) -> ConnectionConfig:
    """Return the named database connection from ``conf/databases(.local).yaml``."""
    blocks = _merged("databases")
    if name not in blocks:
        raise KeyError(f"connection {name!r} not found in conf/databases.yaml")
    return ConnectionConfig(**{key.lower(): value for key, value in blocks[name].items()})


def get_api(name: str) -> ApiConfig:
    """Return the named API config from ``conf/apis(.local).yaml``."""
    blocks = _merged("apis")
    if name not in blocks:
        raise KeyError(f"api {name!r} not found in conf/apis.yaml")
    return ApiConfig(**{key.lower(): value for key, value in blocks[name].items()})
