"""Structured logging — use this instead of ``print`` for anything beyond a quick notebook peek."""

from __future__ import annotations

import logging
import logging.config

import yaml

from core.config import ROOT

_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_CONFIGURED = False


def configure(*, level: str = "INFO") -> None:
    """Load ``conf/logging.yaml`` if present, else a sensible default. Idempotent."""
    global _CONFIGURED
    if _CONFIGURED:
        return
    config_path = ROOT / "conf" / "logging.yaml"
    if config_path.is_file():
        with config_path.open(encoding="utf-8") as handle:
            logging.config.dictConfig(yaml.safe_load(handle))
    else:
        logging.basicConfig(level=level, format=_FORMAT)
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a configured logger (configures logging on first use)."""
    configure()
    return logging.getLogger(name)
