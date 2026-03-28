"""Shared cache locations for generated serializer tests."""

from __future__ import annotations

from pathlib import Path


SCHEMA_CACHE_DIR = Path(__file__).resolve().parents[1] / ".shema_cache"
SCHEMA_CACHE_DIR.mkdir(parents=True, exist_ok=True)


__all__ = ["SCHEMA_CACHE_DIR"]
