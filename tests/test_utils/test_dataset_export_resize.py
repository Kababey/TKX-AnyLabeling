"""Regression coverage for the export-time resize preprocessing pass in
DatasetExportThread. Drives the preprocessing step directly (not through Qt)
so we can assert on the temp artifacts without spinning up a QApplication.
"""

import json
import os
import os.path as osp
import shutil
import tempfile
import unittest

from PIL import Image

from anylabeling.views.labeling.utils.dataset_export import (
    DatasetExportThread,
)


class TestExportResizePreprocess(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="xlabel_export_test_")
        self.img_dir = osp.join(self.tmp, "images")
        self.ann_dir = osp.join(self.tmp, "annotations")
        os.makedirs(self.img_dir)
        os.makedirs(self.ann_dir)

        # 800x400 image with a single 100x100 box in the top-left corner.
        self.img_path = osp.join(self.img_dir, "a.jpg")
        Image.new("RGB", (800, 400), (128, 128, 128)).save(self.img_path)

        self.ann_path = osp.join(self.ann_dir, "a.json")
        with open(self.ann_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "version": "test",
                    "flags": {},
                    "shapes": [
                        {
                            "label": "cat",
                            "shape_type": "rectangle",
                            "points": [
                                [0, 0], [100, 0], [100, 100], [0, 100]
                            ],
                        }
                    ],
                    "imagePath": "a.jpg",
                    "imageData": None,
                    "imageWidth": 800,
                    "imageHeight": 400,
                },
                f,
            )

    def tearDown(self):
        if osp.isdir(self.tmp):
            shutil.rmtree(self.tmp, ignore_errors=True)

    def _make_thread(self, target, mode):
        return DatasetExportThread(
            image_list=[self.img_path],
            partitions=None,
            format_name="YOLO (HBB)",
            output_dir=osp.join(self.tmp, "out"),
            converter=None,
            label_dir_path=self.ann_dir,
            target_resolution=target,
            resize_mode=mode,
        )

    def test_resize_disabled_when_target_missing(self):
        thread = self._make_thread(None, "letterbox")
        self.assertFalse(thread._resize_enabled())

    def test_resize_disabled_when_mode_none(self):
        thread = self._make_thread((320, 320), "none")
        self.assertFalse(thread._resize_enabled())

    def test_letterbox_writes_resized_assets_and_transforms_coords(self):
        thread = self._make_thread((400, 400), "letterbox")
        self.assertTrue(thread._resize_enabled())
        thread._preprocess_resize()
        try:
            self.assertIsNotNone(thread._preprocess_dir)
            resized_img = thread.image_list[0]
            self.assertTrue(osp.isfile(resized_img))
            with Image.open(resized_img) as im:
                self.assertEqual(im.size, (400, 400))

            out_json = osp.join(thread.label_dir_path, "a.json")
            self.assertTrue(osp.isfile(out_json))
            with open(out_json, "r", encoding="utf-8") as f:
                data = json.load(f)

            self.assertEqual(data["imageWidth"], 400)
            self.assertEqual(data["imageHeight"], 400)
            # Letterbox 800x400 -> 400x400 scales by 0.5 and pads height
            # by (400 - 200)//2 = 100 on top. So (0,0) maps to (0, 100)
            # and (100, 100) maps to (50, 150).
            pts = data["shapes"][0]["points"]
            self.assertAlmostEqual(pts[0][0], 0.0, delta=1.0)
            self.assertAlmostEqual(pts[0][1], 100.0, delta=1.0)
            self.assertAlmostEqual(pts[2][0], 50.0, delta=1.0)
            self.assertAlmostEqual(pts[2][1], 150.0, delta=1.0)
        finally:
            thread._cleanup_preprocess()
            self.assertIsNone(thread._preprocess_dir)

    def test_stretch_uses_independent_scale_factors(self):
        thread = self._make_thread((400, 400), "stretch")
        thread._preprocess_resize()
        try:
            out_json = osp.join(thread.label_dir_path, "a.json")
            with open(out_json, "r", encoding="utf-8") as f:
                data = json.load(f)
            pts = data["shapes"][0]["points"]
            # Stretch 800x400 -> 400x400: sx=0.5, sy=1.0
            # (100, 100) -> (50, 100)
            self.assertAlmostEqual(pts[2][0], 50.0, delta=1.0)
            self.assertAlmostEqual(pts[2][1], 100.0, delta=1.0)
        finally:
            thread._cleanup_preprocess()


if __name__ == "__main__":
    unittest.main()
