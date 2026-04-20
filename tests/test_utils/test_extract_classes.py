"""Tests for ExtractClassesThread.run() (called synchronously)."""

import json
import os
import os.path as osp

import pytest

# ExtractClassesThread subclasses QThread, so we need PyQt6 to import it.
# Skip this test module if PyQt6 is unavailable (e.g. headless CI hosts).
pytest.importorskip("PyQt6.QtWidgets")

from anylabeling.views.labeling.widgets.class_manager_dialog import (  # noqa: E402
    ExtractClassesThread,
)


def _write_json(path, payload):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f)


def _write_fake_image(path):
    with open(path, "wb") as f:
        f.write(b"\xff\xd8\xff")  # tiny fake JPEG header


@pytest.fixture
def fixture_dataset(tmp_path):
    images_dir = tmp_path / "images"
    ann_dir = tmp_path / "annotations"
    images_dir.mkdir()
    ann_dir.mkdir()

    _write_fake_image(images_dir / "a_photo.jpg")
    _write_fake_image(images_dir / "b_photo.jpg")

    _write_json(
        ann_dir / "a_photo.json",
        {
            "version": "0.1",
            "flags": {},
            "shapes": [
                {"label": "A", "points": [[0, 0], [1, 1]]},
                {"label": "X", "points": [[2, 2], [3, 3]]},
            ],
            "imagePath": "a_photo.jpg",
        },
    )
    _write_json(
        ann_dir / "b_photo.json",
        {
            "version": "0.1",
            "flags": {},
            "shapes": [
                {"label": "B", "points": [[0, 0], [1, 1]]},
            ],
            "imagePath": "b_photo.jpg",
        },
    )

    return str(images_dir), str(ann_dir), str(tmp_path)


def test_extract_only_selected_class(fixture_dataset, tmp_path):
    src_img, src_ann, _ = fixture_dataset
    dst = tmp_path / "dst"
    dst.mkdir()

    thread = ExtractClassesThread(
        src_images_dir=src_img,
        src_annotations_dir=src_ann,
        dst_root=str(dst),
        selected_labels=["A"],
        keep_only_selected=True,
    )
    thread.run()

    dst_images = dst / "images"
    dst_anns = dst / "annotations"
    assert dst_images.is_dir()
    assert dst_anns.is_dir()

    assert (dst_images / "a_photo.jpg").is_file()
    assert not (dst_images / "b_photo.jpg").exists()

    assert (dst_anns / "a_photo.json").is_file()
    assert not (dst_anns / "b_photo.json").exists()

    with open(dst_anns / "a_photo.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    labels = [s["label"] for s in data["shapes"]]
    assert labels == ["A"], "Only the selected class should survive"

    # Source data must be untouched.
    assert osp.isfile(osp.join(src_img, "a_photo.jpg"))
    assert osp.isfile(osp.join(src_img, "b_photo.jpg"))
    with open(osp.join(src_ann, "a_photo.json"), "r", encoding="utf-8") as f:
        src_data = json.load(f)
    assert len(src_data["shapes"]) == 2


def test_extract_skips_images_without_kept_shapes(fixture_dataset, tmp_path):
    src_img, src_ann, _ = fixture_dataset
    dst = tmp_path / "dst2"
    dst.mkdir()

    thread = ExtractClassesThread(
        src_images_dir=src_img,
        src_annotations_dir=src_ann,
        dst_root=str(dst),
        selected_labels=["B"],
        keep_only_selected=True,
    )
    thread.run()

    dst_anns = dst / "annotations"
    assert (dst_anns / "b_photo.json").is_file()
    assert not (dst_anns / "a_photo.json").exists()
