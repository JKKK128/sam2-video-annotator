"""Configuration loading helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def load_yaml_config(path: Path) -> dict[str, Any]:
    """Load a YAML configuration file."""
    import yaml

    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"config file does not exist: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"config must contain a mapping at top level: {path}")
    return data


def nested_get(config: dict[str, Any], keys: tuple[str, ...], default: Any = None) -> Any:
    """Return a nested config value, or a default when any key is missing."""
    value: Any = config
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return default
        value = value[key]
    return value
