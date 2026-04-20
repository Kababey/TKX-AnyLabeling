"""Dataset import dialog and background worker.

Provides a GUI dialog for importing entire dataset folders in various
formats (YOLO, COCO, VOC, DOTA, XLABEL) with auto-detection, preview,
and background conversion to X-AnyLabeling's native JSON format.
"""

import glob
import json
import os
import os.path as osp
import shutil
import tempfile
import time
import zipfile

from PyQt6 import QtWidgets
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QButtonGroup,
    QGroupBox,
    QHBoxLayout,
    QRadioButton,
    QVBoxLayout,
    QProgressDialog,
)

from anylabeling.views.labeling.label_converter import LabelConverter
from anylabeling.views.labeling.logger import logger
from anylabeling.views.labeling.widgets import Popup
from anylabeling.views.labeling.utils.qt import new_icon_path
from anylabeling.views.labeling.utils.style import (
    get_cancel_btn_style,
    get_export_option_style,
    get_msg_box_style,
    get_ok_btn_style,
    get_progress_dialog_style,
)
from anylabeling.views.labeling.utils.dataset_format_detector import (
    DatasetFormat,
    DatasetStructure,
    detect_dataset_format,
)


# ------------------------------------------------------------------
# Format mapping helpers
# ------------------------------------------------------------------

# Human-readable display names for each format
_FORMAT_DISPLAY_NAMES = {
    DatasetFormat.YOLO_DETECT: "YOLO Detection",
    DatasetFormat.YOLO_SEG: "YOLO Segmentation",
    DatasetFormat.YOLO_OBB: "YOLO OBB",
    DatasetFormat.YOLO_POSE: "YOLO Pose",
    DatasetFormat.COCO_DETECT: "COCO Detection",
    DatasetFormat.COCO_SEG: "COCO Segmentation",
    DatasetFormat.COCO_POSE: "COCO Pose",
    DatasetFormat.VOC_DETECT: "VOC Detection",
    DatasetFormat.VOC_SEG: "VOC Segmentation",
    DatasetFormat.DOTA: "DOTA",
    DatasetFormat.XLABEL: "X-AnyLabeling",
    DatasetFormat.UNKNOWN: "Unknown",
}

# Formats that need a classes file (txt)
_FORMATS_NEEDING_CLASSES_TXT = {
    DatasetFormat.YOLO_DETECT,
    DatasetFormat.YOLO_SEG,
    DatasetFormat.YOLO_OBB,
    DatasetFormat.DOTA,
}

# Formats that need a pose config file (yaml)
_FORMATS_NEEDING_POSE_CFG = {
    DatasetFormat.YOLO_POSE,
}

# COCO-based formats process entire JSON files at once
_COCO_FORMATS = {
    DatasetFormat.COCO_DETECT,
    DatasetFormat.COCO_SEG,
    DatasetFormat.COCO_POSE,
}

# Supported importable formats (dropdown options)
_IMPORTABLE_FORMATS = [
    DatasetFormat.YOLO_DETECT,
    DatasetFormat.YOLO_SEG,
    DatasetFormat.YOLO_OBB,
    DatasetFormat.YOLO_POSE,
    DatasetFormat.COCO_DETECT,
    DatasetFormat.COCO_SEG,
    DatasetFormat.COCO_POSE,
    DatasetFormat.VOC_DETECT,
    DatasetFormat.VOC_SEG,
    DatasetFormat.DOTA,
    DatasetFormat.XLABEL,
]


def _merge_into_current_project(self, active_project, result_dir, image_list):
    """Post-import merge workflow: class mapping, augment project classes,
    assign a split, and refresh the UI. Kept out of ``on_finished`` to
    keep that closure readable and to make the control flow testable.
    """
    from anylabeling.views.labeling.utils.dataset_export import (
        _auto_detect_classes,
    )
    from anylabeling.views.labeling.widgets.class_mapping_dialog import (
        ClassMappingDialog,
    )

    incoming_classes = _auto_detect_classes(image_list, result_dir)
    existing_classes = [c.get("name") for c in active_project.classes or []]
    existing_names = [n for n in existing_classes if n]

    mapping = {name: name for name in incoming_classes}
    if incoming_classes:
        dlg = ClassMappingDialog(
            incoming_classes, existing_names, parent=self
        )
        if dlg.exec() == QtWidgets.QDialog.DialogCode.Accepted:
            mapping = dlg.result_mapping()

    pairs = [(src, dst) for src, dst in mapping.items() if src != dst]
    try:
        _apply_mapping_to_dir(result_dir, pairs)
    except Exception as exc:
        logger.debug("Class mapping rewrite failed: %s", exc)

    try:
        new_final_names = set(mapping.values()) - set(existing_names)
        if new_final_names:
            merged = list(active_project.classes or [])
            for n in sorted(new_final_names):
                merged.append({"name": n, "color": "#888888"})
            self.project_manager.update_classes(active_project, merged)
    except Exception as exc:
        logger.debug("Failed to augment project classes: %s", exc)

    # Reload from the PROJECT images dir so the list includes both the
    # newly imported images and any pre-existing ones.
    self.import_image_folder(
        self.project_manager.get_images_dir(active_project)
    )
    try:
        self.project_manager.update_stats(
            active_project,
            image_count=len(self.image_list),
            annotated_count=sum(
                1 for p in self.image_list
                if osp.isfile(
                    osp.join(
                        result_dir,
                        osp.splitext(osp.basename(p))[0] + ".json",
                    )
                )
            ),
            total_shapes=active_project.stats.get("total_shapes", 0),
        )
    except Exception as exc:
        logger.debug("Failed to update project stats: %s", exc)

    # Offer split assignment for just the newly added images.
    split_mgr = getattr(self, "split_manager", None)
    if split_mgr is not None:
        choices = [
            self.tr("train"),
            self.tr("val"),
            self.tr("test"),
            self.tr("unassigned"),
            self.tr("skip"),
        ]
        picked, ok = QtWidgets.QInputDialog.getItem(
            self,
            self.tr("Assign split"),
            self.tr(
                "Assign the %d newly imported image(s) to which split?"
            ) % len(image_list),
            choices,
            0,
            False,
        )
        if ok and picked and picked != self.tr("skip"):
            split_name = {
                self.tr("train"): "train",
                self.tr("val"): "val",
                self.tr("test"): "test",
                self.tr("unassigned"): "unassigned",
            }.get(picked, picked)
            try:
                for img in image_list:
                    split_mgr.set_partition(img, split_name)
                split_mgr.save_splits()
            except Exception as exc:
                logger.debug("Failed to apply split: %s", exc)


def _apply_mapping_to_dir(ann_dir, pairs):
    """Rewrite shape labels in every JSON under ``ann_dir`` per ``pairs``.

    ``pairs`` is an iterable of ``(src_label, dst_label)``. Files with no
    changed labels are left untouched; corrupt JSONs are skipped so one
    bad file cannot abort the batch mapping.
    """
    if not pairs:
        return
    from anylabeling.views.labeling.utils.project_manager import (
        _atomic_write_json,
    )

    mapping = dict(pairs)
    for jp in glob.glob(osp.join(ann_dir, "*.json")):
        try:
            with open(jp, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(data, dict):
            continue
        changed = False
        for shape in data.get("shapes", []):
            if not isinstance(shape, dict):
                continue
            lbl = shape.get("label")
            if lbl in mapping and mapping[lbl] != lbl:
                shape["label"] = mapping[lbl]
                changed = True
        if changed:
            _atomic_write_json(jp, data)


def _format_display_name(fmt):
    """Return the human-readable name of a DatasetFormat value."""
    return _FORMAT_DISPLAY_NAMES.get(fmt, str(fmt))


# ------------------------------------------------------------------
# ImportThread - background worker
# ------------------------------------------------------------------


class ImportThread(QThread):
    """Background worker that converts dataset annotations to XLABEL JSON.

    Signals:
        progress: (current_index, total_files, percentage)
        finished: (success, error_message, result_directory, image_list)
    """

    progress = pyqtSignal(int, int, int)
    finished = pyqtSignal(bool, str, str, list)

    def __init__(
        self,
        source_path,
        dataset_structure,
        output_dir,
        classes_file=None,
        pose_cfg_file=None,
        format_override=None,
        project_images_dir=None,
        target_resolution=None,
        resize_mode="none",
    ):
        super().__init__()
        self.source_path = source_path
        self.dataset_structure = dataset_structure
        self.output_dir = output_dir
        self.classes_file = classes_file
        self.pose_cfg_file = pose_cfg_file
        self.format_override = format_override
        # Project integration
        self.project_images_dir = project_images_dir
        self.target_resolution = target_resolution  # (w, h) or None
        self.resize_mode = resize_mode  # "letterbox" | "center_crop" | "stretch" | "none"
        self._cancelled = False

    def cancel(self):
        """Request cancellation of the import operation."""
        self._cancelled = True

    def run(self):
        temp_dir = None
        try:
            time.sleep(0.5)

            working_path = self.source_path
            ds = self.dataset_structure
            fmt = self.format_override or ds.format

            # --- Extract ZIP if needed ---
            if zipfile.is_zipfile(self.source_path):
                temp_dir = tempfile.mkdtemp(prefix="xlabel_import_")
                logger.info(
                    "Extracting ZIP to temporary directory: %s", temp_dir
                )
                with zipfile.ZipFile(self.source_path, "r") as zf:
                    zf.extractall(temp_dir)

                # After extraction, re-detect format from the temp dir
                # If there is a single subdirectory, use that as root
                entries = os.listdir(temp_dir)
                if len(entries) == 1:
                    single = osp.join(temp_dir, entries[0])
                    if osp.isdir(single):
                        working_path = single
                    else:
                        working_path = temp_dir
                else:
                    working_path = temp_dir

                ds = detect_dataset_format(working_path)
                if self.format_override:
                    ds.format = self.format_override
                fmt = ds.format

            # --- Prepare output directory ---
            os.makedirs(self.output_dir, exist_ok=True)

            # --- Collect all image paths ---
            all_images = []
            for split_name, image_list in ds.splits.items():
                all_images.extend(image_list)

            if not all_images:
                self.finished.emit(
                    False,
                    "No images found in the dataset.",
                    self.output_dir,
                    [],
                )
                return

            total_files = len(all_images)

            # --- Create converter ---
            converter = self._create_converter(fmt, ds)

            # --- XLABEL format: just copy JSON files ---
            if fmt == DatasetFormat.XLABEL:
                self._import_xlabel(ds, all_images, total_files)

            # --- COCO formats: process entire JSON at once ---
            elif fmt in _COCO_FORMATS:
                self._import_coco(ds, fmt, converter, all_images, total_files)

            # --- Per-file formats: YOLO, VOC, DOTA ---
            else:
                self._import_per_file(
                    ds, fmt, converter, all_images, total_files
                )

        except Exception as e:
            logger.error("Dataset import failed: %s", str(e))
            self.finished.emit(False, str(e), self.output_dir, [])
        finally:
            # Clean up temp directory if we extracted a ZIP
            if temp_dir and osp.exists(temp_dir):
                try:
                    shutil.rmtree(temp_dir)
                except Exception:
                    logger.warning(
                        "Failed to clean up temp directory: %s", temp_dir
                    )

    def _create_converter(self, fmt, ds):
        """Create a LabelConverter with the appropriate classes/pose config."""
        classes_file = self.classes_file or ds.classes_file
        pose_cfg_file = self.pose_cfg_file

        # Guard: LabelConverter expects a plain-text file (one class per
        # line).  If the detected classes_file is actually a YAML
        # (e.g. data.yaml), ignore it so we fall through to ds.classes.
        if classes_file and classes_file.lower().endswith(
            (".yaml", ".yml")
        ):
            classes_file = None

        if fmt in _FORMATS_NEEDING_POSE_CFG:
            if pose_cfg_file:
                return LabelConverter(pose_cfg_file=pose_cfg_file)
            else:
                raise ValueError(
                    "A pose config file (YAML) is required for "
                    f"{_format_display_name(fmt)} format."
                )
        elif fmt in _FORMATS_NEEDING_CLASSES_TXT:
            if classes_file:
                return LabelConverter(classes_file=classes_file)
            elif ds.classes:
                # Write a temporary classes file from detected classes
                tmp_classes = osp.join(
                    tempfile.gettempdir(), "xlabel_import_classes.txt"
                )
                with open(tmp_classes, "w", encoding="utf-8") as f:
                    f.write("\n".join(ds.classes))
                return LabelConverter(classes_file=tmp_classes)
            else:
                raise ValueError(
                    "A classes file (TXT) is required for "
                    f"{_format_display_name(fmt)} format, and no classes "
                    "could be auto-detected."
                )
        elif fmt in _COCO_FORMATS:
            # COCO formats can auto-detect classes from the JSON itself
            if classes_file:
                return LabelConverter(classes_file=classes_file)
            if pose_cfg_file:
                return LabelConverter(pose_cfg_file=pose_cfg_file)
            return LabelConverter()
        else:
            return LabelConverter()

    def _unique_dst_path(self, target_dir, filename):
        """Return a path inside target_dir that does not collide with existing files."""
        dst = osp.join(target_dir, filename)
        if not osp.exists(dst):
            return dst
        base, ext = osp.splitext(filename)
        i = 1
        while True:
            candidate = osp.join(target_dir, f"{base}_{i}{ext}")
            if not osp.exists(candidate):
                return candidate
            i += 1

    def _copy_image_to_project(self, src_image_path):
        """Copy (and optionally resize) an image into the project images dir.

        Returns a tuple (new_image_path, resize_result_or_none).
        When no project_images_dir is set, returns (src_image_path, None).
        """
        if not self.project_images_dir:
            return src_image_path, None

        os.makedirs(self.project_images_dir, exist_ok=True)
        dst = self._unique_dst_path(
            self.project_images_dir, osp.basename(src_image_path)
        )

        # Decide whether to resize
        should_resize = (
            self.target_resolution
            and self.target_resolution[0] > 0
            and self.target_resolution[1] > 0
            and self.resize_mode not in (None, "none", "")
        )
        if should_resize:
            try:
                from anylabeling.views.labeling.utils.image_resizer import (
                    ResizeMode,
                    resize_image,
                )
                mode = ResizeMode(self.resize_mode)
                result = resize_image(
                    src_image_path, dst, self.target_resolution, mode
                )
                if result.success:
                    return dst, result
                logger.warning(
                    "Resize failed for %s: %s. Falling back to copy.",
                    src_image_path, result.error,
                )
            except Exception as e:
                logger.warning("Resize error for %s: %s", src_image_path, e)
        try:
            shutil.copy2(src_image_path, dst)
        except OSError as e:
            logger.warning("Copy failed for %s: %s", src_image_path, e)
            return src_image_path, None
        return dst, None

    def _apply_resize_to_annotation(self, json_path, resize_result):
        """If an annotation JSON exists at json_path, transform its
        coordinates using the provided ResizeResult."""
        if not resize_result or not resize_result.success:
            return
        if not osp.isfile(json_path):
            return
        try:
            from anylabeling.views.labeling.utils.image_resizer import (
                transform_annotation,
            )
            import json as _json
            with open(json_path, "r", encoding="utf-8") as f:
                data = _json.load(f)
            data = transform_annotation(data, resize_result)
            # Also update imagePath to point at the new basename
            data["imagePath"] = osp.basename(resize_result.output_path)
            with open(json_path, "w", encoding="utf-8") as f:
                _json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.warning(
                "Failed to transform annotation %s: %s", json_path, e
            )

    def _import_xlabel(self, ds, all_images, total_files):
        """Import XLABEL format -- copy existing JSON annotation files."""
        copied_images = []
        for i, image_path in enumerate(all_images):
            if self._cancelled:
                break

            # Copy image into project dir if configured
            new_image_path, resize_result = self._copy_image_to_project(
                image_path
            )

            # Use new filename when we copied so the JSON stem matches
            stem = osp.splitext(osp.basename(new_image_path))[0]
            json_name = stem + ".json"
            json_src = osp.join(
                osp.dirname(image_path),
                osp.splitext(osp.basename(image_path))[0] + ".json",
            )
            json_dst = osp.join(self.output_dir, json_name)
            if osp.exists(json_src):
                shutil.copy2(json_src, json_dst)
                if resize_result:
                    self._apply_resize_to_annotation(json_dst, resize_result)

            copied_images.append(new_image_path)
            pct = int((i + 1) / total_files * 100)
            self.progress.emit(i + 1, total_files, pct)

        self.finished.emit(True, "", self.output_dir, copied_images)

    def _import_coco(self, ds, fmt, converter, all_images, total_files):
        """Import COCO format -- process entire JSON files at once."""
        mode_map = {
            DatasetFormat.COCO_DETECT: "rectangle",
            DatasetFormat.COCO_SEG: "polygon",
            DatasetFormat.COCO_POSE: "pose",
        }
        mode = mode_map[fmt]

        # Process each COCO JSON file (one per split)
        coco_jsons = ds.coco_json_paths or {}

        if not coco_jsons:
            # Fallback: look for annotation JSON in annotation_paths
            for split_name, ann_path in ds.annotation_paths.items():
                if osp.isfile(ann_path) and ann_path.endswith(".json"):
                    coco_jsons[split_name] = ann_path

        if not coco_jsons:
            self.finished.emit(
                False,
                "No COCO annotation JSON files found in the dataset.",
                self.output_dir,
                [],
            )
            return

        processed = 0
        for split_name, json_path in coco_jsons.items():
            if self._cancelled:
                break

            logger.info(
                "Processing COCO annotations for split '%s': %s",
                split_name,
                json_path,
            )
            converter.coco_to_custom(
                input_file=json_path,
                output_dir_path=self.output_dir,
                mode=mode,
            )

            # Update progress after each JSON file
            split_images = ds.splits.get(split_name, [])
            processed += len(split_images) if split_images else 1
            pct = min(int(processed / total_files * 100), 100)
            self.progress.emit(processed, total_files, pct)

        # Copy images into project dir (and transform annotations if resized)
        final_images = []
        if self.project_images_dir:
            for image_path in all_images:
                if self._cancelled:
                    break
                new_image_path, resize_result = self._copy_image_to_project(
                    image_path
                )
                final_images.append(new_image_path)

                # Find the generated annotation and fix imagePath / resize
                stem = osp.splitext(osp.basename(image_path))[0]
                json_out = osp.join(self.output_dir, stem + ".json")
                if osp.isfile(json_out):
                    if resize_result:
                        self._apply_resize_to_annotation(
                            json_out, resize_result
                        )
                    else:
                        self._update_annotation_image_path(
                            json_out, osp.basename(new_image_path)
                        )
                    # If the image was renamed to avoid collision,
                    # rename the json to match
                    new_stem = osp.splitext(
                        osp.basename(new_image_path)
                    )[0]
                    if new_stem != stem:
                        renamed = osp.join(
                            self.output_dir, new_stem + ".json"
                        )
                        try:
                            os.replace(json_out, renamed)
                        except OSError:
                            pass
        else:
            final_images = list(all_images)

        self.finished.emit(True, "", self.output_dir, final_images)

    def _import_per_file(self, ds, fmt, converter, all_images, total_files):
        """Import per-file formats: YOLO, VOC, DOTA."""
        imported_images = []

        for i, image_path in enumerate(all_images):
            if self._cancelled:
                break

            # Copy image into project (possibly resized) BEFORE conversion
            # so LabelConverter sees the final image dimensions.
            new_image_path, resize_result = self._copy_image_to_project(
                image_path
            )

            final_basename = osp.basename(new_image_path)
            final_stem = osp.splitext(final_basename)[0]
            output_json = osp.join(self.output_dir, final_stem + ".json")

            # Determine annotation file path (always look in source location)
            ann_file = self._find_annotation_file(ds, image_path, fmt)

            if ann_file and osp.exists(ann_file):
                try:
                    # Convert using the ORIGINAL image so dimensions match
                    # the annotation's coordinate space. We'll transform
                    # the resulting JSON if we resized.
                    self._convert_single_file(
                        converter, fmt, ann_file, output_json, image_path,
                        osp.basename(image_path),
                    )
                    if resize_result:
                        self._apply_resize_to_annotation(
                            output_json, resize_result
                        )
                    else:
                        # Still make sure imagePath points to the new file
                        self._update_annotation_image_path(
                            output_json, final_basename
                        )
                    imported_images.append(new_image_path)
                except Exception as e:
                    logger.warning(
                        "Failed to convert annotation for %s: %s",
                        final_basename,
                        str(e),
                    )
                    imported_images.append(new_image_path)
            else:
                # Image without annotation -- still include it
                imported_images.append(new_image_path)

            pct = int((i + 1) / total_files * 100)
            self.progress.emit(i + 1, total_files, pct)

        self.finished.emit(True, "", self.output_dir, imported_images)

    def _update_annotation_image_path(self, json_path, basename):
        """Update imagePath in an annotation JSON to point to basename."""
        if not osp.isfile(json_path):
            return
        try:
            import json as _json
            with open(json_path, "r", encoding="utf-8") as f:
                data = _json.load(f)
            data["imagePath"] = basename
            with open(json_path, "w", encoding="utf-8") as f:
                _json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception:
            pass

    def _find_annotation_file(self, ds, image_path, fmt):
        """Locate the annotation file corresponding to an image."""
        image_basename = osp.basename(image_path)
        image_stem = osp.splitext(image_basename)[0]
        image_dir = osp.dirname(image_path)

        # Determine the expected annotation extension
        if fmt in (
            DatasetFormat.YOLO_DETECT,
            DatasetFormat.YOLO_SEG,
            DatasetFormat.YOLO_OBB,
            DatasetFormat.YOLO_POSE,
            DatasetFormat.DOTA,
        ):
            ann_ext = ".txt"
        elif fmt in (DatasetFormat.VOC_DETECT, DatasetFormat.VOC_SEG):
            ann_ext = ".xml"
        else:
            ann_ext = ".json"

        ann_filename = image_stem + ann_ext

        # Strategy 1: Check annotation_paths from the dataset structure
        # For each split, check if this image belongs to it
        for split_name, images_in_split in ds.splits.items():
            if image_path in images_in_split:
                ann_dir = ds.annotation_paths.get(split_name, "")
                if ann_dir and osp.isdir(ann_dir):
                    candidate = osp.join(ann_dir, ann_filename)
                    if osp.exists(candidate):
                        return candidate

        # Strategy 2: Check common relative locations
        # labels/ next to images/
        parent = osp.dirname(image_dir)
        dir_name = osp.basename(image_dir)
        for labels_name in ["labels", "labelTxt", "Annotations"]:
            candidate_dir = osp.join(parent, labels_name)
            if osp.isdir(candidate_dir):
                candidate = osp.join(candidate_dir, ann_filename)
                if osp.exists(candidate):
                    return candidate
            # Also check labels/<split>/
            candidate_dir2 = osp.join(parent, labels_name, dir_name)
            if osp.isdir(candidate_dir2):
                candidate = osp.join(candidate_dir2, ann_filename)
                if osp.exists(candidate):
                    return candidate

        # Strategy 3: Same directory as image
        candidate = osp.join(image_dir, ann_filename)
        if osp.exists(candidate):
            return candidate

        return None

    def _convert_single_file(
        self, converter, fmt, ann_file, output_json, image_path,
        image_basename,
    ):
        """Convert a single annotation file to XLABEL JSON."""
        if fmt == DatasetFormat.YOLO_DETECT:
            converter.yolo_to_custom(
                input_file=ann_file,
                output_file=output_json,
                image_file=image_path,
                mode="hbb",
            )
        elif fmt == DatasetFormat.YOLO_SEG:
            converter.yolo_to_custom(
                input_file=ann_file,
                output_file=output_json,
                image_file=image_path,
                mode="seg",
            )
        elif fmt == DatasetFormat.YOLO_OBB:
            converter.yolo_obb_to_custom(
                input_file=ann_file,
                output_file=output_json,
                image_file=image_path,
            )
        elif fmt == DatasetFormat.YOLO_POSE:
            converter.yolo_pose_to_custom(
                input_file=ann_file,
                output_file=output_json,
                image_file=image_path,
            )
        elif fmt == DatasetFormat.VOC_DETECT:
            converter.voc_to_custom(
                input_file=ann_file,
                output_file=output_json,
                image_filename=image_basename,
                mode="rectangle",
            )
        elif fmt == DatasetFormat.VOC_SEG:
            converter.voc_to_custom(
                input_file=ann_file,
                output_file=output_json,
                image_filename=image_basename,
                mode="polygon",
            )
        elif fmt == DatasetFormat.DOTA:
            converter.dota_to_custom(
                input_file=ann_file,
                output_file=output_json,
                image_file=image_path,
            )


# ------------------------------------------------------------------
# import_dataset_dialog - main entry point
# ------------------------------------------------------------------


def import_dataset_dialog(self):
    """Show the dataset import dialog and run the import.

    This function is designed to be called as a method on a
    ``LabelingWidget`` instance (``self`` is the widget).

    Flow:
      1. Source selection (folder or ZIP)
      2. Auto-detection
      3. Confirmation dialog with override options
      4. Background import with progress
      5. Load the imported images in the viewer
    """

    # ----------------------------------------------------------
    # Step 1: Source selection
    # ----------------------------------------------------------
    source_dialog = QtWidgets.QDialog(self)
    source_dialog.setWindowTitle(self.tr("Import Dataset"))
    source_dialog.setMinimumWidth(420)
    source_dialog.setStyleSheet(get_export_option_style())

    src_layout = QVBoxLayout()
    src_layout.setContentsMargins(24, 24, 24, 24)
    src_layout.setSpacing(16)

    src_label = QtWidgets.QLabel(
        self.tr("Select a dataset folder or ZIP file to import:")
    )
    src_label.setWordWrap(True)
    src_layout.addWidget(src_label)

    src_btn_layout = QHBoxLayout()
    src_btn_layout.setSpacing(12)

    folder_btn = QtWidgets.QPushButton(self.tr("Open Folder"))
    folder_btn.setStyleSheet(get_ok_btn_style())
    zip_btn = QtWidgets.QPushButton(self.tr("Open ZIP File"))
    zip_btn.setStyleSheet(get_ok_btn_style())
    cancel_btn = QtWidgets.QPushButton(self.tr("Cancel"))
    cancel_btn.setStyleSheet(get_cancel_btn_style())

    src_btn_layout.addStretch()
    src_btn_layout.addWidget(cancel_btn)
    src_btn_layout.addWidget(folder_btn)
    src_btn_layout.addWidget(zip_btn)
    src_layout.addLayout(src_btn_layout)

    source_dialog.setLayout(src_layout)

    selected_path = [None]  # mutable container for closure

    def on_folder():
        path = QtWidgets.QFileDialog.getExistingDirectory(
            self,
            self.tr("Select Dataset Folder"),
            "",
            QtWidgets.QFileDialog.Option.DontUseNativeDialog,
        )
        if path:
            selected_path[0] = path
            source_dialog.accept()

    def on_zip():
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            self.tr("Select Dataset ZIP File"),
            "",
            self.tr("ZIP Files (*.zip);;All Files (*)"),
        )
        if path:
            selected_path[0] = path
            source_dialog.accept()

    folder_btn.clicked.connect(on_folder)
    zip_btn.clicked.connect(on_zip)
    cancel_btn.clicked.connect(source_dialog.reject)

    if source_dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted:
        return

    source_path = selected_path[0]
    if not source_path:
        return

    # ----------------------------------------------------------
    # Step 2: Auto-detection
    # ----------------------------------------------------------
    detect_path = source_path
    temp_extract_dir = None

    if zipfile.is_zipfile(source_path):
        # For detection purposes, extract to a temp directory
        temp_extract_dir = tempfile.mkdtemp(prefix="xlabel_detect_")
        try:
            with zipfile.ZipFile(source_path, "r") as zf:
                zf.extractall(temp_extract_dir)
            entries = os.listdir(temp_extract_dir)
            if len(entries) == 1:
                single = osp.join(temp_extract_dir, entries[0])
                if osp.isdir(single):
                    detect_path = single
                else:
                    detect_path = temp_extract_dir
            else:
                detect_path = temp_extract_dir
        except Exception as e:
            if temp_extract_dir and osp.exists(temp_extract_dir):
                shutil.rmtree(temp_extract_dir, ignore_errors=True)
            popup = Popup(
                self.tr("Failed to extract ZIP file:\n%s") % str(e),
                self,
                icon=new_icon_path("error", "svg"),
            )
            popup.show_popup(self, position="center")
            return

    try:
        dataset_structure = detect_dataset_format(detect_path)
    except Exception as e:
        logger.error("Dataset format detection failed: %s", str(e))
        popup = Popup(
            self.tr("Failed to detect dataset format:\n%s") % str(e),
            self,
            icon=new_icon_path("error", "svg"),
        )
        popup.show_popup(self, position="center")
        if temp_extract_dir and osp.exists(temp_extract_dir):
            shutil.rmtree(temp_extract_dir, ignore_errors=True)
        return
    finally:
        # Clean up the temp extract used for detection
        if temp_extract_dir and osp.exists(temp_extract_dir):
            shutil.rmtree(temp_extract_dir, ignore_errors=True)

    # ----------------------------------------------------------
    # Step 3: Confirmation dialog
    # ----------------------------------------------------------
    ds = dataset_structure
    detected_fmt = ds.format

    dialog = QtWidgets.QDialog(self)
    dialog.setWindowTitle(self.tr("Import Dataset"))
    dialog.setMinimumWidth(600)
    dialog.setStyleSheet(get_export_option_style())

    main_layout = QVBoxLayout()
    main_layout.setContentsMargins(24, 24, 24, 24)
    main_layout.setSpacing(16)

    # --- Import destination: new project vs. merge into current ---
    active_project = getattr(self, "current_project", None)
    dest_group = QGroupBox(self.tr("Import destination"))
    dest_layout = QVBoxLayout(dest_group)
    dest_layout.setSpacing(6)

    dest_button_group = QButtonGroup(dialog)
    radio_create = QRadioButton(
        self.tr("Create new project (from source on disk)")
    )
    if active_project is not None:
        radio_merge = QRadioButton(
            self.tr("Merge into current project: %s") % active_project.name
        )
    else:
        radio_merge = QRadioButton(self.tr("Merge into current project"))
        radio_merge.setEnabled(False)
        radio_merge.setToolTip(
            self.tr("Open a project first to use merge mode.")
        )
    dest_button_group.addButton(radio_create)
    dest_button_group.addButton(radio_merge)
    dest_layout.addWidget(radio_create)
    dest_layout.addWidget(radio_merge)

    if active_project is not None:
        radio_merge.setChecked(True)
    else:
        radio_create.setChecked(True)

    main_layout.addWidget(dest_group)

    # --- Section 1: Source path (read-only) ---
    src_path_layout = QVBoxLayout()
    src_path_label = QtWidgets.QLabel(self.tr("Source path"))
    src_path_layout.addWidget(src_path_label)

    src_path_edit = QtWidgets.QLineEdit()
    src_path_edit.setText(source_path)
    src_path_edit.setReadOnly(True)
    src_path_layout.addWidget(src_path_edit)
    main_layout.addLayout(src_path_layout)

    # --- Section 2: Detected format with override dropdown ---
    format_layout = QVBoxLayout()
    format_label = QtWidgets.QLabel(self.tr("Detected format"))
    format_layout.addWidget(format_label)

    format_combo = QtWidgets.QComboBox()
    detected_index = 0
    for idx, fmt in enumerate(_IMPORTABLE_FORMATS):
        display = _format_display_name(fmt)
        format_combo.addItem(display, fmt)
        if fmt == detected_fmt:
            detected_index = idx

    # If the detected format is UNKNOWN, add it and select it
    if detected_fmt == DatasetFormat.UNKNOWN:
        format_combo.addItem(
            _format_display_name(DatasetFormat.UNKNOWN), DatasetFormat.UNKNOWN
        )
        detected_index = format_combo.count() - 1

    format_combo.setCurrentIndex(detected_index)
    format_layout.addWidget(format_combo)
    main_layout.addLayout(format_layout)

    # --- Section 3: Structure summary (table) ---
    summary_label = QtWidgets.QLabel(self.tr("Dataset structure"))
    main_layout.addWidget(summary_label)

    split_table = QtWidgets.QTableWidget()
    split_table.setColumnCount(3)
    split_table.setHorizontalHeaderLabels(
        [self.tr("Split"), self.tr("Images"), self.tr("Annotations")]
    )
    split_table.horizontalHeader().setStretchLastSection(True)
    split_table.horizontalHeader().setSectionResizeMode(
        QtWidgets.QHeaderView.ResizeMode.Stretch
    )
    split_table.setEditTriggers(
        QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers
    )
    split_table.setSelectionMode(
        QtWidgets.QAbstractItemView.SelectionMode.NoSelection
    )
    split_table.verticalHeader().setVisible(False)

    splits = ds.splits or {}
    split_table.setRowCount(max(len(splits), 1))

    if splits:
        for row, (split_name, images) in enumerate(splits.items()):
            display_name = split_name if split_name != "_all" else "all"
            split_table.setItem(
                row, 0, QtWidgets.QTableWidgetItem(display_name)
            )
            split_table.setItem(
                row, 1, QtWidgets.QTableWidgetItem(str(len(images)))
            )
            # Count annotations for this split
            ann_dir = ds.annotation_paths.get(split_name, "")
            ann_count = 0
            if ann_dir:
                if osp.isdir(ann_dir):
                    ann_count = len([
                        f for f in os.listdir(ann_dir)
                        if osp.isfile(osp.join(ann_dir, f))
                    ])
                elif osp.isfile(ann_dir):
                    ann_count = 1  # single annotation file (e.g., COCO JSON)
            # For COCO, use coco_json_paths
            if ds.coco_json_paths and split_name in ds.coco_json_paths:
                ann_count = 1  # one JSON per split

            split_table.setItem(
                row, 2, QtWidgets.QTableWidgetItem(str(ann_count))
            )
    else:
        split_table.setItem(0, 0, QtWidgets.QTableWidgetItem("-"))
        split_table.setItem(0, 1, QtWidgets.QTableWidgetItem("0"))
        split_table.setItem(0, 2, QtWidgets.QTableWidgetItem("0"))

    split_table.setMaximumHeight(
        min(35 + 30 * max(len(splits), 1), 200)
    )
    main_layout.addWidget(split_table)

    # Total summary line
    total_images = ds.total_images
    total_annotations = ds.total_annotations
    num_splits = len([s for s in splits if s != "_all"])
    if num_splits == 0:
        num_splits = len(splits)

    summary_text = self.tr(
        "Found %d images in %d split(s), format: %s"
    ) % (total_images, num_splits, _format_display_name(detected_fmt))
    summary_info = QtWidgets.QLabel(summary_text)
    summary_info.setStyleSheet(
        "color: gray; font-style: italic; padding-left: 2px;"
    )
    summary_info.setWordWrap(True)
    main_layout.addWidget(summary_info)

    # --- Section 4: Classes info ---
    classes_layout = QVBoxLayout()
    classes_label_text = self.tr("Classes file")
    if ds.classes:
        classes_label_text += self.tr(" (%d classes detected)") % len(
            ds.classes
        )
    classes_label = QtWidgets.QLabel(classes_label_text)
    classes_layout.addWidget(classes_label)

    classes_path_layout = QHBoxLayout()
    classes_path_layout.setSpacing(8)

    classes_path_edit = QtWidgets.QLineEdit()
    if ds.classes_file:
        classes_path_edit.setText(ds.classes_file)
    classes_path_edit.setPlaceholderText(
        self.tr("Optional - select if auto-detection is wrong")
    )

    def browse_classes_file():
        current_fmt = format_combo.currentData()
        if current_fmt in _FORMATS_NEEDING_POSE_CFG:
            file_filter = self.tr(
                "Pose Config Files (*.yaml *.yml);;All Files (*)"
            )
            title = self.tr("Select a pose config file")
        else:
            file_filter = self.tr("Classes Files (*.txt);;All Files (*)")
            title = self.tr("Select a classes file")

        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            dialog, title, "", file_filter
        )
        if path:
            classes_path_edit.setText(path)

    classes_browse_btn = QtWidgets.QPushButton(self.tr("Browse"))
    classes_browse_btn.clicked.connect(browse_classes_file)
    classes_browse_btn.setStyleSheet(get_cancel_btn_style())

    classes_path_layout.addWidget(classes_path_edit)
    classes_path_layout.addWidget(classes_browse_btn)
    classes_layout.addLayout(classes_path_layout)
    main_layout.addLayout(classes_layout)

    # --- Section 5: Output directory (adapts to destination radio) ---
    output_layout = QVBoxLayout()
    output_label = QtWidgets.QLabel(
        self.tr("Output directory for converted annotations")
    )
    output_layout.addWidget(output_label)

    output_input_layout = QHBoxLayout()
    output_input_layout.setSpacing(8)
    output_edit = QtWidgets.QLineEdit()
    output_edit.setPlaceholderText(self.tr("Select Output Directory"))
    if zipfile.is_zipfile(source_path):
        default_output = osp.join(
            osp.dirname(source_path), "xlabel_annotations"
        )
    else:
        default_output = osp.join(source_path, "xlabel_annotations")
    create_default_output = osp.realpath(default_output)

    project_ann_dir = None
    if active_project is not None:
        project_ann_dir = getattr(self, "project_manager").get_annotations_dir(
            active_project
        )

    def browse_output_dir():
        path = QtWidgets.QFileDialog.getExistingDirectory(
            dialog,
            self.tr("Select Output Directory"),
            output_edit.text(),
            QtWidgets.QFileDialog.Option.DontUseNativeDialog,
        )
        if path:
            output_edit.setText(path)

    output_browse_btn = QtWidgets.QPushButton(self.tr("Browse"))
    output_browse_btn.clicked.connect(browse_output_dir)
    output_browse_btn.setStyleSheet(get_cancel_btn_style())

    output_input_layout.addWidget(output_edit)
    output_input_layout.addWidget(output_browse_btn)
    output_layout.addLayout(output_input_layout)

    def _sync_output_for_destination():
        if radio_merge.isChecked() and project_ann_dir:
            output_edit.setText(project_ann_dir)
            output_edit.setReadOnly(True)
            output_edit.setStyleSheet("background: #f5f5f5;")
            output_browse_btn.setEnabled(False)
            output_label.setText(
                self.tr("Annotations output (project-managed)")
            )
        else:
            output_edit.setReadOnly(False)
            output_edit.setStyleSheet("")
            output_browse_btn.setEnabled(True)
            output_label.setText(
                self.tr("Output directory for converted annotations")
            )
            if not output_edit.text() or output_edit.text() == (
                project_ann_dir or ""
            ):
                output_edit.setText(create_default_output)

    radio_create.toggled.connect(
        lambda _checked: _sync_output_for_destination()
    )
    radio_merge.toggled.connect(
        lambda _checked: _sync_output_for_destination()
    )
    _sync_output_for_destination()

    main_layout.addLayout(output_layout)

    # --- Buttons: Cancel / Import ---
    button_layout = QHBoxLayout()
    button_layout.setContentsMargins(0, 16, 0, 0)
    button_layout.setSpacing(8)

    cancel_button = QtWidgets.QPushButton(self.tr("Cancel"))
    cancel_button.clicked.connect(dialog.reject)
    cancel_button.setStyleSheet(get_cancel_btn_style())

    import_button = QtWidgets.QPushButton(self.tr("Import"))
    import_button.clicked.connect(dialog.accept)
    import_button.setStyleSheet(get_ok_btn_style())

    button_layout.addStretch()
    button_layout.addWidget(cancel_button)
    button_layout.addWidget(import_button)
    main_layout.addLayout(button_layout)

    dialog.setLayout(main_layout)

    result = dialog.exec()
    if not result:
        return

    # --- Gather confirmed settings ---
    chosen_format = format_combo.currentData()
    output_dir = output_edit.text()
    classes_file_path = classes_path_edit.text().strip() or None

    # Determine classes vs pose config
    user_classes_file = None
    user_pose_cfg = None
    if classes_file_path:
        if chosen_format in _FORMATS_NEEDING_POSE_CFG:
            user_pose_cfg = classes_file_path
        else:
            user_classes_file = classes_file_path

    # Validate required files
    if chosen_format in _FORMATS_NEEDING_CLASSES_TXT:
        if not user_classes_file and not ds.classes and not ds.classes_file:
            popup = Popup(
                self.tr(
                    "A classes file is required for %s format.\n"
                    "Please select one and try again."
                )
                % _format_display_name(chosen_format),
                self,
                icon=new_icon_path("warning", "svg"),
            )
            popup.show_popup(self, position="center")
            return

    if chosen_format in _FORMATS_NEEDING_POSE_CFG:
        if not user_pose_cfg:
            popup = Popup(
                self.tr(
                    "A pose config file (YAML) is required for %s format.\n"
                    "Please select one and try again."
                )
                % _format_display_name(chosen_format),
                self,
                icon=new_icon_path("warning", "svg"),
            )
            popup.show_popup(self, position="center")
            return

    # Handle output directory already exists
    if osp.exists(output_dir) and os.listdir(output_dir):
        msg_box = QtWidgets.QMessageBox(self)
        msg_box.setIcon(QtWidgets.QMessageBox.Icon.Warning)
        msg_box.setWindowTitle(self.tr("Output Directory Exists!"))
        msg_box.setText(
            self.tr("The output directory is not empty.\n"
                     "Imported annotations will be added alongside "
                     "existing files.")
        )
        msg_box.setInformativeText(
            self.tr(
                "Click OK to continue importing,\n"
                "or Cancel to choose a different location."
            )
        )
        msg_box.addButton(
            self.tr("OK"), QtWidgets.QMessageBox.ButtonRole.AcceptRole
        )
        cancel_msg_btn = msg_box.addButton(
            self.tr("Cancel"), QtWidgets.QMessageBox.ButtonRole.RejectRole
        )
        msg_box.setStyleSheet(get_msg_box_style())
        msg_box.exec()

        if msg_box.clickedButton() == cancel_msg_btn:
            return
    else:
        os.makedirs(output_dir, exist_ok=True)

    # ----------------------------------------------------------
    # Step 4: Import execution with progress
    # ----------------------------------------------------------
    progress_dialog = QProgressDialog(
        self.tr("Importing dataset..."),
        self.tr("Cancel"),
        0,
        100,
        self,
    )
    progress_dialog.setWindowModality(Qt.WindowModality.WindowModal)
    progress_dialog.setWindowTitle(self.tr("Importing"))
    progress_dialog.setMinimumWidth(500)
    progress_dialog.setMinimumHeight(150)
    progress_dialog.setStyleSheet(
        get_progress_dialog_style(color="#1d1d1f", height=20)
    )

    # Determine format_override only if user changed from detected
    format_override = None
    if chosen_format != detected_fmt:
        format_override = chosen_format

    merge_mode = (
        active_project is not None and radio_merge.isChecked()
    )

    # Project integration: copy images into project dir only in merge mode.
    # When the user picked "Create new project" we leave images in source and
    # fall through to the legacy non-project code path on completion.
    project_images_dir = None
    target_resolution = None
    resize_mode = "none"
    if merge_mode:
        project_images_dir = self.project_manager.get_images_dir(
            active_project
        )
        os.makedirs(project_images_dir, exist_ok=True)
        tr = active_project.settings.get("target_resolution") or [0, 0]
        if (
            isinstance(tr, (list, tuple))
            and len(tr) == 2
            and tr[0] > 0
            and tr[1] > 0
            and active_project.settings.get("auto_resize_new_images", False)
        ):
            target_resolution = (int(tr[0]), int(tr[1]))
            resize_mode = active_project.settings.get(
                "resize_mode", "letterbox"
            )

    self.import_thread = ImportThread(
        source_path=source_path,
        dataset_structure=ds,
        output_dir=output_dir,
        classes_file=user_classes_file,
        pose_cfg_file=user_pose_cfg,
        format_override=format_override,
        project_images_dir=project_images_dir,
        target_resolution=target_resolution,
        resize_mode=resize_mode,
    )

    def on_progress(current, total, percentage):
        progress_dialog.setValue(percentage)
        progress_dialog.setLabelText(
            self.tr("Processing file %d / %d ...") % (current, total)
        )

    def on_finished(success, error_msg, result_dir, image_list):
        progress_dialog.close()
        if success:
            count = len(image_list)
            template = self.tr(
                "Dataset imported successfully!\n"
                "%d images processed.\n"
                "Annotations saved to:\n%s"
            )
            message_text = template % (count, result_dir)
            popup = Popup(
                message_text,
                self,
                icon=new_icon_path("copy-green", "svg"),
            )
            popup.show_popup(self, popup_height=85, position="center")

            # ----------------------------------------------------------
            # Step 5: Load images (+ optional merge-mode bookkeeping)
            # ----------------------------------------------------------
            if image_list:
                # Set the output_dir so annotations are loaded from there
                self.output_dir = result_dir
                if merge_mode:
                    _merge_into_current_project(
                        self, active_project, result_dir, image_list
                    )
                else:
                    # Legacy behavior: load from the source dataset dir
                    first_image_dir = osp.dirname(image_list[0])
                    self.import_image_folder(first_image_dir)
            try:
                self._refresh_annotation_queue()
            except Exception:
                pass
        else:
            message = self.tr(
                "Error occurred while importing dataset:\n%s"
            ) % error_msg
            logger.error(message)
            popup = Popup(
                message,
                self,
                icon=new_icon_path("error", "svg"),
            )
            popup.show_popup(self, position="center")

    self.import_thread.progress.connect(on_progress)
    self.import_thread.finished.connect(on_finished)

    progress_dialog.show()
    self.import_thread.start()

    progress_dialog.canceled.connect(self.import_thread.cancel)
    progress_dialog.canceled.connect(self.import_thread.terminate)
