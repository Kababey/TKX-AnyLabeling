"""Helpers for determining whether an image has been reviewed/annotated."""

import json
import os
import os.path as osp
from typing import List


SIDE_CAR_EMPTY_MARKER = "_xanylabeling_reviewed_empty"


def resolve_label_path(image_path: str, annotations_dir: str) -> str:
    """Return the expected sidecar JSON path for an image."""
    base = osp.splitext(osp.basename(image_path))[0]
    if annotations_dir:
        return osp.join(annotations_dir, base + ".json")
    return osp.join(osp.dirname(image_path), base + ".json")


def is_image_annotated(image_path: str, annotations_dir: str) -> bool:
    """True iff JSON exists AND (shapes non-empty OR reviewed-empty marker set)."""
    json_path = resolve_label_path(image_path, annotations_dir)
    if not osp.isfile(json_path):
        return False
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return False
    if not isinstance(data, dict):
        return False
    shapes = data.get("shapes")
    if isinstance(shapes, list) and len(shapes) > 0:
        return True
    return bool(data.get(SIDE_CAR_EMPTY_MARKER, False))


def mark_reviewed_empty(image_path: str, annotations_dir: str) -> str:
    """Write a sidecar JSON marking the image as reviewed-empty. Returns the path."""
    # Lazy import to avoid circular dependency with project_manager.
    from anylabeling.views.labeling.utils.project_manager import _atomic_write_json

    json_path = resolve_label_path(image_path, annotations_dir)
    parent = osp.dirname(json_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    payload = {
        "version": "reviewed-empty",
        "flags": {},
        "shapes": [],
        "imagePath": osp.basename(image_path),
        "imageData": None,
        SIDE_CAR_EMPTY_MARKER: True,
    }
    _atomic_write_json(json_path, payload)
    return json_path


def scan_unlabeled(image_paths: List[str], annotations_dir: str) -> List[str]:
    """Return the subset of image_paths that are NOT annotated."""
    return [p for p in image_paths if not is_image_annotated(p, annotations_dir)]
