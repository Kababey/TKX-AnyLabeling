import json
import os
import tempfile
import unittest

import yaml
from PIL import Image

from anylabeling.views.labeling.label_converter import LabelConverter


class TestLabelConverterPoseConfig(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil

        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def _write_pose_cfg(self, data):
        cfg_path = os.path.join(self.temp_dir, "pose.yaml")
        with open(cfg_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(data, f, sort_keys=False)
        return cfg_path

    def test_missing_has_visible_defaults_to_true(self):
        cfg_path = self._write_pose_cfg(
            {"classes": {"person": ["nose", "left_eye"]}}
        )

        converter = LabelConverter(pose_cfg_file=cfg_path)

        self.assertTrue(converter.has_visible)
        self.assertEqual(converter.classes, ["person"])

    def test_explicit_has_visible_false_is_respected(self):
        cfg_path = self._write_pose_cfg(
            {
                "has_visible": False,
                "classes": {"person": ["nose", "left_eye"]},
            }
        )

        converter = LabelConverter(pose_cfg_file=cfg_path)

        self.assertFalse(converter.has_visible)

    def test_missing_classes_raises_value_error(self):
        cfg_path = self._write_pose_cfg({"has_visible": True})

        with self.assertRaises(ValueError):
            LabelConverter(pose_cfg_file=cfg_path)


class TestResolveImageDims(unittest.TestCase):
    """Regression coverage for the imageWidth/imageHeight fallback used by
    every custom_to_<fmt> exporter."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        import shutil

        if os.path.exists(self.tmp):
            shutil.rmtree(self.tmp)

    def _write_image(self, name, size):
        path = os.path.join(self.tmp, name)
        Image.new("RGB", size, (0, 0, 0)).save(path)
        return path

    def _write_ann(self, name, payload):
        path = os.path.join(self.tmp, name)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f)
        return path

    def test_uses_explicit_dims_when_present(self):
        ann = self._write_ann("a.json", {
            "imageWidth": 320,
            "imageHeight": 240,
            "imagePath": "missing.jpg",
            "shapes": [],
        })
        data = LabelConverter.read_json(ann)
        w, h = LabelConverter._resolve_image_dims(data, ann)
        self.assertEqual((w, h), (320, 240))

    def test_falls_back_to_image_when_keys_missing(self):
        img = self._write_image("photo.png", (640, 480))
        ann = self._write_ann("photo.json", {
            "imagePath": "photo.png",
            "shapes": [],
        })
        data = LabelConverter.read_json(ann)
        w, h = LabelConverter._resolve_image_dims(data, ann)
        self.assertEqual((w, h), (640, 480))
        # The recovered values should be patched back into the dict so a
        # subsequent re-save persists them.
        self.assertEqual(data["imageWidth"], 640)
        self.assertEqual(data["imageHeight"], 480)

    def test_falls_back_via_sibling_when_image_path_unset(self):
        self._write_image("only.png", (100, 50))
        ann = self._write_ann("only.json", {"shapes": []})
        data = LabelConverter.read_json(ann)
        w, h = LabelConverter._resolve_image_dims(data, ann)
        self.assertEqual((w, h), (100, 50))

    def test_falls_back_when_dims_are_invalid(self):
        img = self._write_image("p.jpg", (50, 70))
        ann = self._write_ann("p.json", {
            "imageWidth": -1,
            "imageHeight": 0,
            "imagePath": "p.jpg",
            "shapes": [],
        })
        data = LabelConverter.read_json(ann)
        w, h = LabelConverter._resolve_image_dims(data, ann)
        self.assertEqual((w, h), (50, 70))

    def test_raises_when_image_unavailable(self):
        ann = self._write_ann("ghost.json", {
            "imagePath": "nope.jpg",
            "shapes": [],
        })
        data = LabelConverter.read_json(ann)
        with self.assertRaises(KeyError):
            LabelConverter._resolve_image_dims(data, ann)
