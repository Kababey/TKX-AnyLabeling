"""Review/triage state for labeled vs. unlabeled images.

Status model (per image, keyed by basename):
  - "todo"     : no annotation file yet
  - "labeled"  : annotation file with >= 1 shape
  - "negative" : annotation file with 0 shapes (intentional negative sample)

"approved" is an orthogonal flag persisted in the project config manifest
(handled by ProjectManager). Only the approved bit is non-derivable; the
base status is always computed from the annotation file on disk.
"""

import json
import os.path as osp

TODO = "todo"
LABELED = "labeled"
NEGATIVE = "negative"
APPROVED = "approved"

STATUSES = (TODO, LABELED, NEGATIVE)


def image_key(image_path: str) -> str:
    """Stable manifest key for an image (its basename)."""
    return osp.basename(image_path)


def label_path_for(image_path: str, output_dir: str = "") -> str:
    """Resolve the annotation .json path for an image.

    Mirrors the convention used elsewhere in label_widget: when
    ``output_dir`` is set, annotations live there with the image's
    basename; otherwise the .json sits next to the image.
    """
    base = osp.splitext(osp.basename(image_path))[0] + ".json"
    if output_dir:
        return osp.join(output_dir, base)
    return osp.splitext(image_path)[0] + ".json"


def _shape_count(label_path: str) -> int:
    """Return number of shapes, or -1 if the file is missing/unreadable."""
    if not osp.isfile(label_path):
        return -1
    try:
        with open(label_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError):
        return -1
    shapes = data.get("shapes") if isinstance(data, dict) else None
    if not isinstance(shapes, list):
        return -1
    return len(shapes)


def base_status(image_path: str, output_dir: str = "") -> str:
    n = _shape_count(label_path_for(image_path, output_dir))
    if n < 0:
        return TODO
    return LABELED if n > 0 else NEGATIVE


def classify(image_path: str, output_dir: str, approved_keys) -> dict:
    """Return ``{"base": <status>, "approved": <bool>}`` for an image."""
    return {
        "base": base_status(image_path, output_dir),
        "approved": image_key(image_path) in approved_keys,
    }
