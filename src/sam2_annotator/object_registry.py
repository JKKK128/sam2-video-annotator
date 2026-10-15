"""Per-video local object registry validation and persistence."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

ObjectStatus = Literal["active", "out_of_view", "finished"]
ALLOWED_STATUSES = {"active", "out_of_view", "finished"}
MAX_ACTIVE_OBJECTS = 2


def load_registry(path: Path) -> dict[str, Any]:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"object registry does not exist: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("object registry must be a JSON object")
    validate_registry(data)
    return data


def save_registry(path: Path, registry: dict[str, Any]) -> None:
    validate_registry(registry)
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(registry, indent=2) + "\n", encoding="utf-8")


def validate_registry(registry: dict[str, Any]) -> None:
    if not isinstance(registry.get("video"), str) or not registry["video"]:
        raise ValueError("registry video must be a non-empty string")
    if not isinstance(registry.get("frames_dir"), str) or not registry["frames_dir"]:
        raise ValueError("registry frames_dir must be a non-empty string")
    objects = registry.get("objects")
    invalid_objects = registry.get("invalid_objects", [])
    if not isinstance(objects, list):
        raise ValueError("registry objects must be a list")
    if not isinstance(invalid_objects, list):
        raise ValueError("registry invalid_objects must be a list")
    obj_ids: set[int] = set()
    names: set[str] = set()
    active_global_ids: set[str] = set()
    active_count = 0
    for index, obj in enumerate(objects):
        if not isinstance(obj, dict):
            raise ValueError(f"registry object {index} must be an object")
        obj_id = obj.get("obj_id")
        if not isinstance(obj_id, int) or obj_id <= 0:
            raise ValueError(f"object {index} obj_id must be a positive integer")
        if obj_id in obj_ids:
            raise ValueError(f"duplicate obj_id in registry: {obj_id}")
        obj_ids.add(obj_id)
        name = obj.get("name")
        global_drone_id = obj.get("global_drone_id")
        class_id = obj.get("class_id")
        if not isinstance(name, str) or not name:
            raise ValueError(f"object {obj_id} name must be a non-empty string")
        if name in names:
            raise ValueError(f"duplicate object name in registry: {name}")
        names.add(name)
        if not isinstance(global_drone_id, str) or not global_drone_id:
            raise ValueError(f"object {obj_id} global_drone_id must be a non-empty string")
        if not isinstance(class_id, int) or class_id < 0:
            raise ValueError(f"object {obj_id} class_id must be a non-negative integer")
        first_visible_frame = obj.get("first_visible_frame")
        if not isinstance(first_visible_frame, int) or first_visible_frame < 0:
            raise ValueError(f"object {obj_id} first_visible_frame must be non-negative")
        if not isinstance(obj.get("validated"), bool):
            raise ValueError(f"object {obj_id} validated must be a boolean")
        status = obj.get("status")
        if status not in ALLOWED_STATUSES:
            raise ValueError(f"object {obj_id} status must be one of {sorted(ALLOWED_STATUSES)}")
        if status == "active":
            active_count += 1
            if global_drone_id in active_global_ids:
                raise ValueError(f"duplicate active global_drone_id in registry: {global_drone_id}")
            active_global_ids.add(global_drone_id)
    if active_count > MAX_ACTIVE_OBJECTS:
        raise ValueError(f"a video can have at most {MAX_ACTIVE_OBJECTS} active objects, got {active_count}")
    invalid_ids: set[int] = set()
    for index, obj in enumerate(invalid_objects):
        if not isinstance(obj, dict):
            raise ValueError(f"invalid object {index} must be an object")
        obj_id = obj.get("obj_id")
        if not isinstance(obj_id, int) or obj_id <= 0:
            raise ValueError(f"invalid object {index} obj_id must be a positive integer")
        if obj_id in obj_ids or obj_id in invalid_ids:
            raise ValueError(f"invalid obj_id conflicts with registry object: {obj_id}")
        invalid_ids.add(obj_id)
        if not isinstance(obj.get("invalid_reason"), str) or not obj["invalid_reason"]:
            raise ValueError(f"invalid object {obj_id} invalid_reason must be non-empty")


def get_object(registry: dict[str, Any], obj_id: int) -> dict[str, Any]:
    validate_registry(registry)
    for obj in registry["objects"]:
        if obj["obj_id"] == obj_id:
            return obj
    raise KeyError(f"obj_id {obj_id} not found in registry")


def upsert_object(registry: dict[str, Any], obj_id: int, name: str, global_drone_id: str, class_id: int, class_name: str, first_visible_frame: int, status: ObjectStatus = "active", validated: bool = True) -> dict[str, Any]:
    if obj_id <= 0 or first_visible_frame < 0:
        raise ValueError("obj_id must be positive and first_visible_frame must be non-negative")
    objects = registry.setdefault("objects", [])
    registry.setdefault("invalid_objects", [])
    if not isinstance(objects, list):
        raise ValueError("registry objects must be a list")
    value = {"obj_id": obj_id, "name": name, "global_drone_id": global_drone_id, "class_id": class_id, "class_name": class_name, "first_visible_frame": first_visible_frame, "status": status, "validated": validated}
    for obj in objects:
        if obj.get("obj_id") == obj_id:
            obj.update(value)
            validate_registry(registry)
            return obj
    objects.append(value)
    validate_registry(registry)
    return value
