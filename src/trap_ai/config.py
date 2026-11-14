"""Configuration loading helpers."""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any


def resolve_config_path(root: Path, config_path: Path) -> Path:
    """Resolve config paths relative to the project root."""

    return config_path if config_path.is_absolute() else root / config_path


def load_toml(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        return tomllib.load(handle)
