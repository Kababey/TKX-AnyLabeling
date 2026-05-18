"""Keep a project's stats and its YOLO ``labels/`` folder in sync.

X-AnyLabeling projects store annotations as XLabel ``.json`` files in
``annotations/``. The augmentation engine, however, consumes YOLO-seg
``.txt`` files from ``labels/``. Without reconciliation, newly labelled
images never reach augmentation and the Project Manager shows stale
counts.

This module provides two crash-safe helpers:

  * :func:`recount_project` - scan disk for accurate image / annotated
    / shape counts.
  * :func:`sync_project_yolo_labels` - regenerate ``labels/*.txt`` (plus
    ``classes.txt`` / ``data.yaml``) from the current ``annotations/``
    so augmentation always sees every labelled image.
"""

import json
import os
import os.path as osp

from anylabeling.views.labeling.logger import logger

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")


def _list_images(images_dir: str):
    if not osp.isdir(images_dir):
        return []
    return sorted(
        f for f in os.listdir(images_dir)
        if f.lower().endswith(IMAGE_EXTS)
    )


def _read_json(path: str):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def recount_project(images_dir: str, annotations_dir: str) -> dict:
    """Return accurate ``{image_count, annotated_count, total_shapes}``.

    *annotated* = an image whose ``.json`` exists and has >= 1 shape.
    Negative samples (empty json) count as images but not annotated.
    """
    images = _list_images(images_dir)
    image_count = len(images)
    annotated = 0
    total_shapes = 0
    for img in images:
        stem = osp.splitext(img)[0]
        jp = osp.join(annotations_dir, stem + ".json")
        if not osp.isfile(jp):
            continue
        data = _read_json(jp)
        if not isinstance(data, dict):
            continue
        shapes = data.get("shapes")
        n = len(shapes) if isinstance(shapes, list) else 0
        if n > 0:
            annotated += 1
            total_shapes += n
    return {
        "image_count": image_count,
        "annotated_count": annotated,
        "total_shapes": total_shapes,
    }


def _collect_labels_in_annotations(annotations_dir: str) -> list:
    """Every distinct shape label across all annotation jsons (sorted)."""
    seen = []
    seen_set = set()
    if not osp.isdir(annotations_dir):
        return seen
    for fn in sorted(os.listdir(annotations_dir)):
        if not fn.lower().endswith(".json"):
            continue
        data = _read_json(osp.join(annotations_dir, fn))
        if not isinstance(data, dict):
            continue
        for sh in data.get("shapes", []) or []:
            lbl = sh.get("label")
            if lbl and lbl not in seen_set:
                seen_set.add(lbl)
                seen.append(lbl)
    return seen


def _resolve_class_order(project_classes, annotations_dir: str) -> list:
    """Project class order first, then any extra labels found on disk.

    Keeping the project order stable means existing YOLO indices don't
    shift when a new class appears.
    """
    order = []
    for c in project_classes or []:
        name = c.get("name") if isinstance(c, dict) else str(c)
        if name and name not in order:
            order.append(name)
    for lbl in _collect_labels_in_annotations(annotations_dir):
        if lbl not in order:
            order.append(lbl)
    return order


def sync_project_yolo_labels(
    project_root: str,
    images_subdir: str = "images",
    annotations_subdir: str = "annotations",
    project_classes=None,
) -> dict:
    """Regenerate ``project_root/labels/*.txt`` from ``annotations/``.

    Also (re)writes ``classes.txt`` and ``data.yaml`` at the project
    root so the folder is a valid YOLO-seg dataset for augmentation.

    Returns ``{images, converted, empty, classes, labels_dir}`` or
    ``{"error": ...}``. Never raises.
    """
    try:
        from anylabeling.views.labeling.label_converter import LabelConverter

        images_dir = osp.join(project_root, images_subdir)
        annotations_dir = osp.join(project_root, annotations_subdir)
        labels_dir = osp.join(project_root, "labels")
        os.makedirs(labels_dir, exist_ok=True)

        images = _list_images(images_dir)
        if not images:
            return {"error": f"No images in {images_dir}"}

        classes = _resolve_class_order(project_classes, annotations_dir)

        # Persist classes.txt + data.yaml at the project root.
        classes_path = osp.join(project_root, "classes.txt")
        try:
            with open(classes_path, "w", encoding="utf-8") as f:
                f.write("\n".join(classes) + ("\n" if classes else ""))
        except OSError as exc:
            logger.warning("project_sync: classes.txt write failed: %s", exc)

        try:
            with open(
                osp.join(project_root, "data.yaml"), "w", encoding="utf-8"
            ) as f:
                f.write(
                    f"path: {osp.abspath(project_root)}\n"
                    "train: images\n"
                    "val: images\n"
                    f"nc: {len(classes)}\n"
                    f"names: {list(classes)!r}\n"
                )
        except OSError as exc:
            logger.warning("project_sync: data.yaml write failed: %s", exc)

        converter = LabelConverter(classes_file=classes_path)
        # Belt-and-suspenders: ensure converter has the full class list
        # even if reading classes.txt missed something.
        if classes and getattr(converter, "classes", None) != classes:
            converter.classes = list(classes)

        converted = 0
        empty = 0
        for img in images:
            stem = osp.splitext(img)[0]
            src_json = osp.join(annotations_dir, stem + ".json")
            dst_txt = osp.join(labels_dir, stem + ".txt")
            try:
                if osp.isfile(src_json):
                    is_empty = converter.custom_to_yolo(
                        src_json, dst_txt, "seg", False
                    )
                    if is_empty:
                        empty += 1
                    else:
                        converted += 1
                else:
                    # No annotation yet -> empty label (negative sample).
                    open(dst_txt, "w", encoding="utf-8").close()
                    empty += 1
            except Exception as exc:  # one bad file must not abort sync
                logger.warning(
                    "project_sync: convert failed for %s: %s", src_json, exc
                )
                # Still emit an empty label so counts stay consistent.
                try:
                    open(dst_txt, "w", encoding="utf-8").close()
                except OSError:
                    pass
                empty += 1

        return {
            "images": len(images),
            "converted": converted,
            "empty": empty,
            "classes": classes,
            "labels_dir": labels_dir,
        }
    except Exception as exc:
        logger.warning("project_sync: sync failed: %s", exc)
        return {"error": f"{type(exc).__name__}: {exc}"}
