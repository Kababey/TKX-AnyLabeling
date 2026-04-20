"""Tests for anylabeling.views.labeling.utils.annotation_status."""

import json
import os
import os.path as osp

import pytest

from anylabeling.views.labeling.utils.annotation_status import (
    SIDE_CAR_EMPTY_MARKER,
    is_image_annotated,
    mark_reviewed_empty,
    resolve_label_path,
    scan_unlabeled,
)


@pytest.fixture
def img_tree(tmp_path):
    images_dir = tmp_path / "images"
    ann_dir = tmp_path / "annotations"
    images_dir.mkdir()
    ann_dir.mkdir()
    img = images_dir / "photo.jpg"
    img.write_bytes(b"\xff\xd8\xff")  # tiny fake jpeg header
    return str(img), str(ann_dir)


def _write_ann(ann_dir, img_path, payload):
    base = osp.splitext(osp.basename(img_path))[0]
    path = osp.join(ann_dir, base + ".json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f)
    return path


def test_resolve_label_path_uses_annotations_dir(tmp_path):
    img = str(tmp_path / "foo.png")
    ann_dir = str(tmp_path / "ann")
    assert resolve_label_path(img, ann_dir) == osp.join(ann_dir, "foo.json")


def test_missing_json_is_not_annotated(img_tree):
    img, ann = img_tree
    assert is_image_annotated(img, ann) is False


def test_json_with_shapes_is_annotated(img_tree):
    img, ann = img_tree
    _write_ann(ann, img, {"shapes": [{"label": "cat"}]})
    assert is_image_annotated(img, ann) is True


def test_empty_shapes_without_marker_is_not_annotated(img_tree):
    img, ann = img_tree
    _write_ann(ann, img, {"shapes": []})
    assert is_image_annotated(img, ann) is False


def test_empty_shapes_with_marker_is_annotated(img_tree):
    img, ann = img_tree
    _write_ann(ann, img, {"shapes": [], SIDE_CAR_EMPTY_MARKER: True})
    assert is_image_annotated(img, ann) is True


def test_corrupt_json_is_not_annotated(img_tree):
    img, ann = img_tree
    base = osp.splitext(osp.basename(img))[0]
    with open(osp.join(ann, base + ".json"), "w", encoding="utf-8") as f:
        f.write("{ this is not json")
    assert is_image_annotated(img, ann) is False


def test_mark_reviewed_empty_roundtrip(img_tree):
    img, ann = img_tree
    assert is_image_annotated(img, ann) is False
    path = mark_reviewed_empty(img, ann)
    assert osp.isfile(path)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert data["shapes"] == []
    assert data[SIDE_CAR_EMPTY_MARKER] is True
    assert is_image_annotated(img, ann) is True


def test_scan_unlabeled_filters_correctly(tmp_path):
    ann = str(tmp_path / "ann")
    os.makedirs(ann, exist_ok=True)
    imgs = []
    for i in range(3):
        p = tmp_path / f"i{i}.jpg"
        p.write_bytes(b"\xff")
        imgs.append(str(p))
    _write_ann(ann, imgs[0], {"shapes": [{"label": "a"}]})
    _write_ann(ann, imgs[2], {"shapes": [], SIDE_CAR_EMPTY_MARKER: True})
    # imgs[1] has no sidecar
    result = scan_unlabeled(imgs, ann)
    assert result == [imgs[1]]


def test_mark_reviewed_empty_creates_annotations_dir(tmp_path):
    img = tmp_path / "i.jpg"
    img.write_bytes(b"\xff")
    ann = tmp_path / "new_ann_dir"
    mark_reviewed_empty(str(img), str(ann))
    assert ann.is_dir()
