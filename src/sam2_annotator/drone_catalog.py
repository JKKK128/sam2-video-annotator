"""Loading and validation for the dataset-wide fixed drone identity catalog."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_catalog(path: Path) -> dict[str, dict[str, Any]]:
    """Return catalog entries keyed by ``global_drone_id``."""
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"drone catalog does not exist: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    entries = data.get("drones") if isinstance(data, dict) else None
    if not isinstance(entries, list):
        raise ValueError("drone catalog drones must be a list")
    catalog: dict[str, dict[str, Any]] = {}
    class_ids: set[int] = set()
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise ValueError(f"catalog entry {index} must be an object")
        drone_id = entry.get("global_drone_id")
        class_name = entry.get("class_name")
        class_id = entry.get("class_id")
        if not isinstance(drone_id, str) or not drone_id:
            raise ValueError(f"catalog entry {index} global_drone_id must be non-empty")
        if not isinstance(class_name, str) or not class_name:
            raise ValueError(f"catalog entry {drone_id} class_name must be non-empty")
        if not isinstance(class_id, int) or class_id < 0:
            raise ValueError(f"catalog entry {drone_id} class_id must be a non-negative integer")
        if drone_id in catalog:
            raise ValueError(f"duplicate global_drone_id in catalog: {drone_id}")
        if class_id in class_ids:
            raise ValueError(f"duplicate class_id in catalog: {class_id}")
        catalog[drone_id] = {"global_drone_id": drone_id, "class_id": class_id, "class_name": class_name}
        class_ids.add(class_id)
    if not catalog:
        raise ValueError("drone catalog must contain at least one drone")
    return catalog


def get_catalog_entry(catalog: dict[str, dict[str, Any]], global_drone_id: str) -> dict[str, Any]:
    try:
        return catalog[global_drone_id]
    except KeyError as exc:
        raise ValueError(f"undefined global_drone_id: {global_drone_id}") from exc
