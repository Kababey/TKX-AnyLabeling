import json
import os
import os.path as osp
import pathlib
import shutil
import tempfile
import time
import yaml
import zipfile

from PyQt6 import QtWidgets
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QButtonGroup,
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QProgressDialog,
    QRadioButton,
    QVBoxLayout,
)

from anylabeling.views.labeling.label_converter import LabelConverter
from anylabeling.views.labeling.logger import logger
from anylabeling.views.labeling.utils.qt import new_icon_path
from anylabeling.views.labeling.utils.style import (
    get_cancel_btn_style,
    get_export_option_style,
    get_msg_box_style,
    get_ok_btn_style,
    get_progress_dialog_style,
)
from anylabeling.views.labeling.widgets import Popup


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FORMAT_MAP = {
    "YOLO (HBB)": ("custom_to_yolo", "hbb"),
    "YOLO (OBB)": ("custom_to_yolo", "obb"),
    "YOLO (Segmentation)": ("custom_to_yolo", "seg"),
    "YOLO (Pose)": ("custom_to_yolo", "pose"),
    "COCO (Detection)": ("custom_to_coco", "rectangle"),
    "COCO (Segmentation)": ("custom_to_coco", "polygon"),
    "COCO (Pose)": ("custom_to_coco", "pose"),
    "VOC (Detection)": ("custom_to_voc", "rectangle"),
    "VOC (Segmentation)": ("custom_to_voc", "polygon"),
    "DOTA": ("custom_to_dota", None),
    "Mask": ("custom_to_mask", None),
}

# Formats that accept (but no longer require) a classes file.
# When no file is provided the classes are auto-detected from annotations.
FORMATS_NEEDING_CLASSES = {
    "YOLO (HBB)",
    "YOLO (OBB)",
    "YOLO (Segmentation)",
    "YOLO (Pose)",
    "COCO (Detection)",
    "COCO (Segmentation)",
    "COCO (Pose)",
    "DOTA",
}

# Formats that need a yaml config specifically (pose variants).
FORMATS_NEEDING_YAML = {
    "YOLO (Pose)",
    "COCO (Pose)",
}

# Mask format requires a JSON color-map file instead of classes.txt.
FORMAT_MASK = "Mask"


def _is_yolo_format(fmt_name):
    return fmt_name.startswith("YOLO")


def _is_coco_format(fmt_name):
    return fmt_name.startswith("COCO")


def _is_voc_format(fmt_name):
    return fmt_name.startswith("VOC")


def _auto_detect_classes(image_list, label_dir):
    """Scan XLABEL annotation JSON files and collect all unique class labels.

    Args:
        image_list: List of absolute image file paths.
        label_dir: Directory where annotation JSONs are stored (the app's
                   ``output_dir``).  Falls back to each image's own
                   directory when a JSON is not found in *label_dir*.

    Returns:
        A sorted list of unique class-name strings.
    """
    classes = set()
    for image_file in image_list:
        base = os.path.splitext(os.path.basename(image_file))[0]
        json_path = os.path.join(label_dir, base + ".json")
        if not os.path.isfile(json_path):
            # Fallback: same directory as the image.
            json_path = os.path.join(
                os.path.dirname(image_file), base + ".json"
            )
        if os.path.isfile(json_path):
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                for shape in data.get("shapes", []):
                    label = shape.get("label", "")
                    if label:
                        classes.add(label)
            except Exception:
                pass
    return sorted(classes)


# ---------------------------------------------------------------------------
# DatasetExportThread
# ---------------------------------------------------------------------------


class DatasetExportThread(QThread):
    """Background worker that exports the full dataset with directory
    structure, optional splits, and optional ZIP archive creation.

    Signals:
        progress(int, int): (current_index, total_count)
        finished(bool, str): (success, error_message)
    """

    progress = pyqtSignal(int, int)
    finished = pyqtSignal(bool, str)

    def __init__(
        self,
        image_list,
        partitions,
        format_name,
        output_dir,
        converter,
        label_dir_path,
        save_images=False,
        skip_empty=False,
        create_zip=False,
        color_map=None,
    ):
        """
        Args:
            image_list: List of absolute image file paths.
            partitions: dict mapping basename -> split name
                        (e.g. {"img001.jpg": "train"}).
                        ``None`` means no split structure.
            format_name: Key into FORMAT_MAP.
            output_dir: Root output directory.
            converter: A fully-initialised LabelConverter instance.
            label_dir_path: Directory containing XLABEL .json files.
            save_images: Whether to copy source images into the output.
            skip_empty: Whether to skip images with no annotations.
            create_zip: Whether to create a ZIP of the output directory.
            color_map: Mapping table dict for Mask export (required when
                       format_name == "Mask").
        """
        super().__init__()
        self.image_list = image_list
        self.partitions = partitions
        self.format_name = format_name
        self.output_dir = output_dir
        self.converter = converter
        self.label_dir_path = label_dir_path
        self.save_images = save_images
        self.skip_empty = skip_empty
        self.create_zip = create_zip
        self.color_map = color_map
        self._cancelled = False

    # ---- public helpers ----

    def cancel(self):
        self._cancelled = True

    # ---- internal helpers ----

    def _get_label_file(self, image_file):
        """Resolve the XLABEL JSON file for a given image."""
        basename = osp.basename(image_file)
        label_name = osp.splitext(basename)[0] + ".json"
        # Prefer label_dir_path (output_dir of the app).
        candidate = osp.join(self.label_dir_path, label_name)
        if osp.exists(candidate):
            return candidate
        # Fallback: same directory as the image.
        candidate = osp.join(osp.dirname(image_file), label_name)
        if osp.exists(candidate):
            return candidate
        return None

    def _split_for(self, image_file):
        """Return the split name for an image, or None if no splits."""
        if self.partitions is None:
            return None
        return self.partitions.get(osp.basename(image_file), "unassigned")

    # ---- directory structure builders ----

    def _make_yolo_dirs(self, splits):
        """Create YOLO directory structure and return path helpers."""
        if splits:
            for s in splits:
                os.makedirs(
                    osp.join(self.output_dir, "images", s), exist_ok=True
                )
                os.makedirs(
                    osp.join(self.output_dir, "labels", s), exist_ok=True
                )
        else:
            os.makedirs(
                osp.join(self.output_dir, "images"), exist_ok=True
            )
            os.makedirs(
                osp.join(self.output_dir, "labels"), exist_ok=True
            )

    def _make_coco_dirs(self, splits):
        """Create COCO directory structure."""
        os.makedirs(
            osp.join(self.output_dir, "annotations"), exist_ok=True
        )
        if self.save_images:
            if splits:
                for s in splits:
                    os.makedirs(
                        osp.join(self.output_dir, s), exist_ok=True
                    )
            else:
                os.makedirs(
                    osp.join(self.output_dir, "images"), exist_ok=True
                )

    def _make_voc_dirs(self, splits):
        """Create VOC directory structure."""
        os.makedirs(
            osp.join(self.output_dir, "Annotations"), exist_ok=True
        )
        if self.save_images:
            os.makedirs(
                osp.join(self.output_dir, "JPEGImages"), exist_ok=True
            )
        if splits:
            os.makedirs(
                osp.join(self.output_dir, "ImageSets", "Main"), exist_ok=True
            )

    def _make_generic_dirs(self, splits):
        """Create generic directory structure for DOTA / Mask / others."""
        if splits:
            for s in splits:
                os.makedirs(
                    osp.join(self.output_dir, "labels", s), exist_ok=True
                )
                if self.save_images:
                    os.makedirs(
                        osp.join(self.output_dir, "images", s), exist_ok=True
                    )
        else:
            os.makedirs(
                osp.join(self.output_dir, "labels"), exist_ok=True
            )
            if self.save_images:
                os.makedirs(
                    osp.join(self.output_dir, "images"), exist_ok=True
                )

    # ---- YOLO data.yaml generation ----

    def _write_yolo_data_yaml(self, splits):
        """Write a data.yaml file for YOLO training."""
        data = {}
        if splits:
            for s in splits:
                data[s] = f"./images/{s}"
        else:
            data["train"] = "./images"

        data["nc"] = len(self.converter.classes)
        data["names"] = list(self.converter.classes)

        yaml_path = osp.join(self.output_dir, "data.yaml")
        with open(yaml_path, "w", encoding="utf-8") as f:
            yaml.dump(data, f, default_flow_style=False, allow_unicode=True)

    def _write_yolo_classes_txt(self):
        """Write a classes.txt file for YOLO."""
        classes_path = osp.join(self.output_dir, "classes.txt")
        with open(classes_path, "w", encoding="utf-8") as f:
            for cls in self.converter.classes:
                f.write(f"{cls}\n")

    # ---- per-format export runners ----

    def _export_yolo(self):
        """Export dataset in YOLO format."""
        method_name, mode = FORMAT_MAP[self.format_name]
        splits = self._get_active_splits()
        self._make_yolo_dirs(splits)
        total = len(self.image_list)

        for i, image_file in enumerate(self.image_list):
            if self._cancelled:
                return
            self.progress.emit(i, total)

            basename = osp.basename(image_file)
            stem = osp.splitext(basename)[0]
            split = self._split_for(image_file)
            label_file = self._get_label_file(image_file)

            # Determine output paths.
            if splits and split:
                label_dst = osp.join(
                    self.output_dir, "labels", split, stem + ".txt"
                )
                image_dst = osp.join(
                    self.output_dir, "images", split, basename
                )
            else:
                label_dst = osp.join(
                    self.output_dir, "labels", stem + ".txt"
                )
                image_dst = osp.join(
                    self.output_dir, "images", basename
                )

            # Skip unassigned images when using splits.
            if splits and split == "unassigned":
                continue

            # Convert annotation.
            src_file = label_file if label_file else ""
            if not osp.exists(src_file):
                if not self.skip_empty:
                    pathlib.Path(label_dst).touch()
                    if self.save_images:
                        shutil.copy2(image_file, image_dst)
                continue

            is_empty = self.converter.custom_to_yolo(
                src_file, label_dst, mode, self.skip_empty
            )

            if self.skip_empty and is_empty:
                # Remove label file if it was created.
                if osp.exists(label_dst):
                    os.remove(label_dst)
                continue

            if self.save_images:
                shutil.copy2(image_file, image_dst)

        # Write metadata files.
        self._write_yolo_classes_txt()
        self._write_yolo_data_yaml(splits)
        self.progress.emit(total, total)

    def _export_coco(self):
        """Export dataset in COCO format.

        COCO conversion is handled as a batch operation by the converter.
        When splits are active, we call the converter once per split with
        the subset of images belonging to that split.
        """
        _, mode = FORMAT_MAP[self.format_name]
        splits = self._get_active_splits()
        self._make_coco_dirs(splits)
        total = len(self.image_list)

        if splits:
            # Group images by split.
            split_images = {s: [] for s in splits}
            for image_file in self.image_list:
                split = self._split_for(image_file)
                if split and split != "unassigned" and split in split_images:
                    split_images[split].append(image_file)

            processed = 0
            for split_name, images in split_images.items():
                if self._cancelled:
                    return
                if not images:
                    continue

                # Create a temporary annotations directory for this split.
                split_ann_dir = osp.join(self.output_dir, "annotations")

                # The COCO converter writes a single JSON to the output
                # path.  We call it with a temporary output path and then
                # rename the generated file to include the split name.
                tmp_output = osp.join(self.output_dir, "_tmp_coco_export")
                os.makedirs(tmp_output, exist_ok=True)

                self.converter.custom_to_coco(
                    images, self.label_dir_path, tmp_output, mode
                )

                # Determine the generated filename and rename.
                if mode == "rectangle":
                    src_name = "coco_detection.json"
                    dst_name = f"instances_{split_name}.json"
                elif mode == "polygon":
                    src_name = "coco_instance_segmentation.json"
                    dst_name = f"instances_{split_name}.json"
                elif mode == "pose":
                    src_name = "coco_keypoints.json"
                    dst_name = f"person_keypoints_{split_name}.json"
                else:
                    src_name = "coco_detection.json"
                    dst_name = f"instances_{split_name}.json"

                src_path = osp.join(tmp_output, src_name)
                dst_path = osp.join(split_ann_dir, dst_name)
                if osp.exists(src_path):
                    shutil.move(src_path, dst_path)
                # Clean up temp dir.
                shutil.rmtree(tmp_output, ignore_errors=True)

                # Copy images if requested.
                if self.save_images:
                    for img in images:
                        if self._cancelled:
                            return
                        img_dst = osp.join(
                            self.output_dir, split_name, osp.basename(img)
                        )
                        shutil.copy2(img, img_dst)
                        processed += 1
                        self.progress.emit(processed, total)
                else:
                    processed += len(images)
                    self.progress.emit(processed, total)
        else:
            # No splits -- single batch export.
            ann_dir = osp.join(self.output_dir, "annotations")
            self.converter.custom_to_coco(
                self.image_list, self.label_dir_path, ann_dir, mode
            )
            if self.save_images:
                for i, img in enumerate(self.image_list):
                    if self._cancelled:
                        return
                    img_dst = osp.join(
                        self.output_dir, "images", osp.basename(img)
                    )
                    shutil.copy2(img, img_dst)
                    self.progress.emit(i + 1, total)
            else:
                self.progress.emit(total, total)

        self.progress.emit(total, total)

    def _export_voc(self):
        """Export dataset in Pascal VOC format."""
        _, mode = FORMAT_MAP[self.format_name]
        splits = self._get_active_splits()
        self._make_voc_dirs(splits)
        total = len(self.image_list)

        # Track which images belong to which split for ImageSets.
        split_lists = {s: [] for s in splits} if splits else {}

        for i, image_file in enumerate(self.image_list):
            if self._cancelled:
                return
            self.progress.emit(i, total)

            basename = osp.basename(image_file)
            stem = osp.splitext(basename)[0]
            split = self._split_for(image_file)
            label_file = self._get_label_file(image_file)

            # Skip unassigned when using splits.
            if splits and split == "unassigned":
                continue

            # Track split membership.
            if splits and split in split_lists:
                split_lists[split].append(stem)

            # VOC annotations always go into Annotations/ (flat).
            ann_dst = osp.join(self.output_dir, "Annotations", stem + ".xml")

            src_file = label_file if label_file else ""
            if not osp.exists(src_file):
                if not self.skip_empty:
                    # Create an empty XML for consistency.
                    if self.save_images:
                        shutil.copy2(
                            image_file,
                            osp.join(
                                self.output_dir, "JPEGImages", basename
                            ),
                        )
                continue

            is_empty = self.converter.custom_to_voc(
                image_file, src_file, ann_dst, mode, self.skip_empty
            )

            if self.skip_empty and is_empty:
                if osp.exists(ann_dst):
                    os.remove(ann_dst)
                continue

            if self.save_images:
                shutil.copy2(
                    image_file,
                    osp.join(self.output_dir, "JPEGImages", basename),
                )

        # Write ImageSets files.
        if splits:
            imagesets_dir = osp.join(
                self.output_dir, "ImageSets", "Main"
            )
            for split_name, stems in split_lists.items():
                if stems:
                    filepath = osp.join(imagesets_dir, f"{split_name}.txt")
                    with open(filepath, "w", encoding="utf-8") as f:
                        for s in stems:
                            f.write(f"{s}\n")

        self.progress.emit(total, total)

    def _export_dota(self):
        """Export dataset in DOTA format."""
        splits = self._get_active_splits()
        self._make_generic_dirs(splits)
        total = len(self.image_list)

        for i, image_file in enumerate(self.image_list):
            if self._cancelled:
                return
            self.progress.emit(i, total)

            basename = osp.basename(image_file)
            stem = osp.splitext(basename)[0]
            split = self._split_for(image_file)
            label_file = self._get_label_file(image_file)

            if splits and split == "unassigned":
                continue

            if splits and split:
                label_dst = osp.join(
                    self.output_dir, "labels", split, stem + ".txt"
                )
                image_dst = osp.join(
                    self.output_dir, "images", split, basename
                )
            else:
                label_dst = osp.join(
                    self.output_dir, "labels", stem + ".txt"
                )
                image_dst = osp.join(
                    self.output_dir, "images", basename
                )

            src_file = label_file if label_file else ""
            if not osp.exists(src_file):
                if not self.skip_empty:
                    pathlib.Path(label_dst).touch()
                    if self.save_images:
                        shutil.copy2(image_file, image_dst)
                continue

            self.converter.custom_to_dota(src_file, label_dst)

            if self.save_images:
                shutil.copy2(image_file, image_dst)

        self.progress.emit(total, total)

    def _export_mask(self):
        """Export dataset in Mask format."""
        splits = self._get_active_splits()
        self._make_generic_dirs(splits)
        total = len(self.image_list)

        for i, image_file in enumerate(self.image_list):
            if self._cancelled:
                return
            self.progress.emit(i, total)

            basename = osp.basename(image_file)
            stem = osp.splitext(basename)[0]
            split = self._split_for(image_file)
            label_file = self._get_label_file(image_file)

            if splits and split == "unassigned":
                continue

            if splits and split:
                label_dst = osp.join(
                    self.output_dir, "labels", split, stem + ".png"
                )
                image_dst = osp.join(
                    self.output_dir, "images", split, basename
                )
            else:
                label_dst = osp.join(
                    self.output_dir, "labels", stem + ".png"
                )
                image_dst = osp.join(
                    self.output_dir, "images", basename
                )

            src_file = label_file if label_file else ""
            if not osp.exists(src_file):
                continue

            self.converter.custom_to_mask(
                src_file, label_dst, self.color_map
            )

            if self.save_images:
                shutil.copy2(image_file, image_dst)

        self.progress.emit(total, total)

    # ---- helpers ----

    def _get_active_splits(self):
        """Return a list of active split names, or empty list if no
        partitions are configured."""
        if self.partitions is None:
            return []
        # Collect unique split names actually used, excluding unassigned.
        used = set(self.partitions.values())
        used.discard("unassigned")
        # Maintain a stable order.
        ordered = []
        for name in ("train", "val", "test"):
            if name in used:
                ordered.append(name)
        # Include any custom splits that aren't in the standard set.
        for name in sorted(used):
            if name not in ordered:
                ordered.append(name)
        return ordered

    def _create_zip_archive(self):
        """Create a ZIP archive of the output directory."""
        zip_path = self.output_dir.rstrip("/\\") + ".zip"
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for root, _dirs, files in os.walk(self.output_dir):
                for file in files:
                    abs_path = osp.join(root, file)
                    arc_name = osp.relpath(abs_path, osp.dirname(self.output_dir))
                    zf.write(abs_path, arc_name)
        return zip_path

    # ---- main entry ----

    def run(self):
        try:
            time.sleep(0.3)

            if _is_yolo_format(self.format_name):
                self._export_yolo()
            elif _is_coco_format(self.format_name):
                self._export_coco()
            elif _is_voc_format(self.format_name):
                self._export_voc()
            elif self.format_name == "DOTA":
                self._export_dota()
            elif self.format_name == FORMAT_MASK:
                self._export_mask()
            else:
                raise ValueError(
                    f"Unsupported export format: {self.format_name}"
                )

            if self._cancelled:
                self.finished.emit(False, "Export cancelled by user.")
                return

            zip_path = None
            if self.create_zip:
                zip_path = self._create_zip_archive()

            self.finished.emit(True, zip_path or "")
        except Exception as e:
            logger.error("Dataset export failed: %s", e, exc_info=True)
            self.finished.emit(False, str(e))


# ---------------------------------------------------------------------------
# Dialog entry-point (bound to LabelingWidget as `self`)
# ---------------------------------------------------------------------------


def export_dataset_dialog(self):
    """Show the unified dataset export dialog.

    ``self`` is expected to be the ``LabelingWidget`` instance.
    """
    # ---- pre-check ----
    if not self.may_continue():
        return

    if not self.filename:
        popup = Popup(
            self.tr("Please load an image folder before proceeding!"),
            self,
            icon=new_icon_path("warning", "svg"),
        )
        popup.show_popup(self, position="center")
        return

    # ---- build dialog ----
    dialog = QtWidgets.QDialog(self)
    dialog.setWindowTitle(self.tr("Export Dataset"))
    dialog.setMinimumWidth(560)
    dialog.setStyleSheet(get_export_option_style())

    layout = QVBoxLayout()
    layout.setContentsMargins(24, 24, 24, 24)
    layout.setSpacing(16)

    # -- 1. Format selection --
    format_label = QtWidgets.QLabel(self.tr("Export format"))
    layout.addWidget(format_label)

    format_combo = QComboBox()
    for fmt_name in FORMAT_MAP:
        format_combo.addItem(fmt_name)
    layout.addWidget(format_combo)

    # -- 2. Classes / config file (optional for most formats) --
    classes_group = QGroupBox(self.tr("Classes / Config file"))
    classes_layout = QVBoxLayout()
    classes_layout.setSpacing(8)

    classes_hint = QtWidgets.QLabel(
        self.tr(
            "Classes are auto-detected from annotations when no file is "
            "selected.\nA .yaml config file is required for Pose formats.\n"
            "For Mask format, select a JSON color-map file."
        )
    )
    classes_hint.setWordWrap(True)
    classes_hint.setStyleSheet("color: gray; font-style: italic;")
    classes_layout.addWidget(classes_hint)

    classes_path_layout = QHBoxLayout()
    classes_path_layout.setSpacing(8)

    classes_edit = QtWidgets.QLineEdit()
    classes_edit.setPlaceholderText(
        self.tr(
            "Optional - auto-detect from annotations "
            "(or select classes.txt / .yaml / .json)"
        )
    )

    def browse_classes_file():
        fmt = format_combo.currentText()
        if fmt in FORMATS_NEEDING_YAML:
            filt = self.tr("Config Files (*.yaml *.yml);;All Files (*)")
            title = self.tr("Select pose config file")
        elif fmt == FORMAT_MASK:
            filt = self.tr("JSON Files (*.json);;All Files (*)")
            title = self.tr("Select color-map file")
        else:
            filt = self.tr("Classes Files (*.txt);;All Files (*)")
            title = self.tr("Select classes file")

        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            dialog, title, "", filt
        )
        if path:
            classes_edit.setText(path)

    classes_button = QtWidgets.QPushButton(self.tr("Browse"))
    classes_button.clicked.connect(browse_classes_file)
    classes_button.setStyleSheet(get_cancel_btn_style())

    classes_path_layout.addWidget(classes_edit)
    classes_path_layout.addWidget(classes_button)
    classes_layout.addLayout(classes_path_layout)
    classes_group.setLayout(classes_layout)
    layout.addWidget(classes_group)

    # -- 3. Split handling --
    split_group = QGroupBox(self.tr("Split handling"))
    split_layout = QVBoxLayout()
    split_layout.setSpacing(8)

    split_btn_group = QButtonGroup(dialog)
    radio_use_splits = QRadioButton(
        self.tr("Use current partition assignments")
    )
    radio_single = QRadioButton(
        self.tr("Export all as single folder (no split structure)")
    )
    split_btn_group.addButton(radio_use_splits, 0)
    split_btn_group.addButton(radio_single, 1)

    # Enable split radio only if SplitManager is available with splits.
    has_splits = False
    if hasattr(self, "split_manager") and self.split_manager is not None:
        try:
            has_splits = self.split_manager.has_splits()
        except Exception:
            has_splits = False

    radio_use_splits.setEnabled(has_splits)
    if has_splits:
        radio_use_splits.setChecked(True)
    else:
        radio_single.setChecked(True)
        radio_use_splits.setToolTip(
            self.tr(
                "No partition assignments found. "
                "Use the Split Manager to assign images first."
            )
        )

    split_layout.addWidget(radio_use_splits)
    split_layout.addWidget(radio_single)
    split_group.setLayout(split_layout)
    layout.addWidget(split_group)

    # -- 4. Output directory --
    output_label = QtWidgets.QLabel(self.tr("Output directory"))
    layout.addWidget(output_label)

    output_path_layout = QHBoxLayout()
    output_path_layout.setSpacing(8)

    output_edit = QtWidgets.QLineEdit()
    default_output = osp.realpath(
        osp.join(osp.dirname(self.filename), "..", "dataset_export")
    )
    output_edit.setText(default_output)
    output_edit.setPlaceholderText(self.tr("Select output directory"))

    def browse_output_dir():
        path = QtWidgets.QFileDialog.getExistingDirectory(
            dialog,
            self.tr("Select Output Directory"),
            output_edit.text(),
            QtWidgets.QFileDialog.Option.DontUseNativeDialog,
        )
        if path:
            output_edit.setText(path)

    output_button = QtWidgets.QPushButton(self.tr("Browse"))
    output_button.clicked.connect(browse_output_dir)
    output_button.setStyleSheet(get_cancel_btn_style())

    output_path_layout.addWidget(output_edit)
    output_path_layout.addWidget(output_button)
    layout.addLayout(output_path_layout)

    # -- 5. Options --
    options_group = QGroupBox(self.tr("Options"))
    options_layout = QVBoxLayout()
    options_layout.setSpacing(8)

    chk_images = QtWidgets.QCheckBox(self.tr("Include images"))
    chk_images.setChecked(False)
    options_layout.addWidget(chk_images)

    chk_skip_empty = QtWidgets.QCheckBox(self.tr("Skip empty labels"))
    chk_skip_empty.setChecked(False)
    options_layout.addWidget(chk_skip_empty)

    chk_zip = QtWidgets.QCheckBox(self.tr("Create ZIP archive"))
    chk_zip.setChecked(False)
    options_layout.addWidget(chk_zip)

    options_group.setLayout(options_layout)
    layout.addWidget(options_group)

    # -- 6. Buttons --
    button_layout = QHBoxLayout()
    button_layout.setContentsMargins(0, 16, 0, 0)
    button_layout.setSpacing(8)

    cancel_button = QtWidgets.QPushButton(self.tr("Cancel"))
    cancel_button.clicked.connect(dialog.reject)
    cancel_button.setStyleSheet(get_cancel_btn_style())

    export_button = QtWidgets.QPushButton(self.tr("Export"))
    export_button.clicked.connect(dialog.accept)
    export_button.setStyleSheet(get_ok_btn_style())

    button_layout.addStretch()
    button_layout.addWidget(cancel_button)
    button_layout.addWidget(export_button)
    layout.addLayout(button_layout)

    dialog.setLayout(layout)

    # ---- show dialog ----
    result = dialog.exec()
    if not result:
        return

    # ---- collect user choices ----
    format_name = format_combo.currentText()
    classes_path = classes_edit.text().strip()
    use_splits = radio_use_splits.isChecked() and has_splits
    output_dir = output_edit.text().strip()
    save_images = chk_images.isChecked()
    skip_empty = chk_skip_empty.isChecked()
    create_zip = chk_zip.isChecked()

    # ---- validate ----
    if not output_dir:
        popup = Popup(
            self.tr("Please specify an output directory."),
            self,
            icon=new_icon_path("warning", "svg"),
        )
        popup.show_popup(self, position="center")
        return

    # Validate classes/config file.
    color_map = None
    converter = None

    # ---- image list (needed early for auto-detection) ----
    image_list = self.image_list if self.image_list else [self.filename]

    # ---- label directory (needed early for auto-detection) ----
    label_dir_path = osp.dirname(self.filename)
    if self.output_dir:
        label_dir_path = self.output_dir

    if format_name in FORMATS_NEEDING_CLASSES:
        if classes_path and not osp.isfile(classes_path):
            popup = Popup(
                self.tr("Classes file not found:\n%s") % classes_path,
                self,
                icon=new_icon_path("error", "svg"),
            )
            popup.show_popup(self, popup_height=65, position="center")
            return

        if format_name in FORMATS_NEEDING_YAML:
            # Pose formats always require a YAML config file.
            if not classes_path:
                popup = Popup(
                    self.tr(
                        "A YAML config file is required for %s format.\n"
                        "Please select a .yaml file."
                    )
                    % format_name,
                    self,
                    icon=new_icon_path("warning", "svg"),
                )
                popup.show_popup(self, popup_height=65, position="center")
                return
            try:
                converter = LabelConverter(pose_cfg_file=classes_path)
            except Exception as e:
                logger.error(
                    "Failed to load pose config: %s: %s", classes_path, e
                )
                popup = Popup(
                    self.tr("Invalid pose config file:\n%s") % str(e),
                    self,
                    icon=new_icon_path("error", "svg"),
                )
                popup.show_popup(self, popup_height=65, position="center")
                return
        elif classes_path:
            # User provided a classes file explicitly -- use it.
            converter = LabelConverter(classes_file=classes_path)
        else:
            # Auto-detect classes from annotation JSON files.
            detected = _auto_detect_classes(image_list, label_dir_path)
            if not detected:
                popup = Popup(
                    self.tr(
                        "No classes could be auto-detected from the "
                        "annotations.\nPlease select a classes file "
                        "manually."
                    ),
                    self,
                    icon=new_icon_path("warning", "svg"),
                )
                popup.show_popup(self, popup_height=65, position="center")
                return

            logger.info(
                "Auto-detected %d classes: %s", len(detected), detected
            )

            # Write a temporary classes.txt so LabelConverter can load it.
            tmp_classes_fd, tmp_classes_path = tempfile.mkstemp(
                suffix=".txt", prefix="xlabel_classes_"
            )
            try:
                with os.fdopen(tmp_classes_fd, "w", encoding="utf-8") as f:
                    for cls in detected:
                        f.write(f"{cls}\n")
                converter = LabelConverter(classes_file=tmp_classes_path)
            finally:
                # Clean up the temp file after the converter has loaded it.
                try:
                    os.remove(tmp_classes_path)
                except OSError:
                    pass
    elif format_name == FORMAT_MASK:
        if not classes_path:
            popup = Popup(
                self.tr(
                    "A color-map JSON file is required for Mask format.\n"
                    "Please select a .json file."
                ),
                self,
                icon=new_icon_path("warning", "svg"),
            )
            popup.show_popup(self, popup_height=65, position="center")
            return

        if not osp.isfile(classes_path):
            popup = Popup(
                self.tr("Color-map file not found:\n%s") % classes_path,
                self,
                icon=new_icon_path("error", "svg"),
            )
            popup.show_popup(self, popup_height=65, position="center")
            return

        try:
            with open(classes_path, "r", encoding="utf-8") as f:
                color_map = json.load(f)
        except Exception as e:
            logger.error(
                "Failed to load color-map: %s: %s", classes_path, e
            )
            popup = Popup(
                self.tr("Invalid color-map file:\n%s") % str(e),
                self,
                icon=new_icon_path("error", "svg"),
            )
            popup.show_popup(self, popup_height=65, position="center")
            return

        converter = LabelConverter()
    else:
        # Formats that do not need a classes file.
        converter = LabelConverter()

    # ---- handle output directory ----
    if osp.exists(output_dir):
        msg_box = QtWidgets.QMessageBox(self)
        msg_box.setIcon(QtWidgets.QMessageBox.Icon.Warning)
        msg_box.setWindowTitle(self.tr("Output Directory Exists!"))
        msg_box.setText(
            self.tr("Directory already exists. Choose an action:")
        )
        msg_box.setInformativeText(
            self.tr(
                "- Yes    - Merge with existing files\n"
                "- No     - Delete existing directory\n"
                "- Cancel - Abort export"
            )
        )

        msg_box.addButton(
            self.tr("Yes"), QtWidgets.QMessageBox.ButtonRole.YesRole
        )
        no_button = msg_box.addButton(
            self.tr("No"), QtWidgets.QMessageBox.ButtonRole.NoRole
        )
        cancel_btn = msg_box.addButton(
            self.tr("Cancel"), QtWidgets.QMessageBox.ButtonRole.RejectRole
        )
        msg_box.setStyleSheet(get_msg_box_style())
        msg_box.exec()

        clicked = msg_box.clickedButton()
        if clicked == no_button:
            shutil.rmtree(output_dir)
            os.makedirs(output_dir)
        elif clicked == cancel_btn:
            return
    else:
        os.makedirs(output_dir)

    # ---- build partitions dict ----
    partitions = None
    if use_splits:
        partitions = {}
        all_splits = self.split_manager.get_all_splits()
        for split_name, filenames in all_splits.items():
            for fname in filenames:
                partitions[fname] = split_name

    # ---- progress dialog ----
    total = len(image_list)
    progress_dialog = QProgressDialog(
        self.tr("Exporting dataset..."),
        self.tr("Cancel"),
        0,
        total,
        self,
    )
    progress_dialog.setWindowModality(Qt.WindowModality.WindowModal)
    progress_dialog.setWindowTitle(self.tr("Export Progress"))
    progress_dialog.setMinimumWidth(500)
    progress_dialog.setMinimumHeight(150)
    progress_dialog.setStyleSheet(
        get_progress_dialog_style(color="#1d1d1f", height=20)
    )

    # ---- create and start thread ----
    self._dataset_export_thread = DatasetExportThread(
        image_list=image_list,
        partitions=partitions,
        format_name=format_name,
        output_dir=output_dir,
        converter=converter,
        label_dir_path=label_dir_path,
        save_images=save_images,
        skip_empty=skip_empty,
        create_zip=create_zip,
        color_map=color_map,
    )

    def on_progress(current, total_count):
        progress_dialog.setMaximum(total_count)
        progress_dialog.setValue(current)

    def on_finished(success, message):
        progress_dialog.close()
        if success:
            if message and message.endswith(".zip"):
                text = self.tr(
                    "Dataset exported successfully!\n"
                    "Results saved to:\n%s\n\n"
                    "ZIP archive created:\n%s"
                ) % (output_dir, message)
            else:
                text = self.tr(
                    "Dataset exported successfully!\n"
                    "Results saved to:\n%s"
                ) % output_dir
            popup = Popup(
                text,
                self,
                icon=new_icon_path("copy-green", "svg"),
            )
            popup.show_popup(self, popup_height=65, position="center")
        else:
            error_text = self.tr(
                "Error exporting dataset:\n%s"
            ) % message
            logger.error("Dataset export error: %s", message)
            popup = Popup(
                error_text,
                self,
                icon=new_icon_path("error", "svg"),
            )
            popup.show_popup(self, position="center")

    self._dataset_export_thread.progress.connect(on_progress)
    self._dataset_export_thread.finished.connect(on_finished)

    progress_dialog.canceled.connect(self._dataset_export_thread.cancel)

    progress_dialog.show()
    self._dataset_export_thread.start()
