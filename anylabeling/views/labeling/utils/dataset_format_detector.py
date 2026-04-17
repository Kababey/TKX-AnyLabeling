"""Dataset format auto-detection module.

Examines a folder structure and determines what annotation format
a dataset uses (YOLO, COCO, VOC, DOTA, or native X-AnyLabeling)
along with its split layout and class definitions.
"""

import json
import os
import os.path as osp
import random
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple

import yaml

from anylabeling.views.labeling.logger import logger

IMG_FORMATS = [".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"]

# Split directory aliases grouped by canonical name
_SPLIT_ALIASES = {
    "train": ["train", "training", "train2017", "train2014"],
    "val": ["val", "valid", "validation", "val2017", "val2014"],
    "test": ["test", "testing", "test2017", "test2014"],
}


# ------------------------------------------------------------------
# Public data types
# ------------------------------------------------------------------

class DatasetFormat(Enum):
    """Supported annotation formats."""

    YOLO_DETECT = auto()
    YOLO_SEG = auto()
    YOLO_OBB = auto()
    YOLO_POSE = auto()
    COCO_DETECT = auto()
    COCO_SEG = auto()
    COCO_POSE = auto()
    VOC_DETECT = auto()
    VOC_SEG = auto()
    DOTA = auto()
    XLABEL = auto()
    UNKNOWN = auto()


@dataclass
class DatasetStructure:
    """Result of dataset format detection.

    Attributes:
        format: Detected annotation format.
        root_path: Absolute path to the dataset root.
        splits: Maps split name to list of image file paths.
        annotation_paths: Maps split name to annotation directory or file.
        classes: Detected class names.
        classes_file: Path to the classes file, if found.
        coco_json_paths: For COCO format, maps split name to JSON path.
        total_images: Total number of images across all splits.
        total_annotations: Total number of annotation files/entries.
    """

    format: DatasetFormat = DatasetFormat.UNKNOWN
    root_path: str = ""
    splits: Dict[str, List[str]] = field(default_factory=dict)
    annotation_paths: Dict[str, str] = field(default_factory=dict)
    classes: List[str] = field(default_factory=list)
    classes_file: Optional[str] = None
    coco_json_paths: Optional[Dict[str, str]] = None
    total_images: int = 0
    total_annotations: int = 0


# ------------------------------------------------------------------
# Main public entry point
# ------------------------------------------------------------------

def detect_dataset_format(path: str) -> DatasetStructure:
    """Detect the annotation format and structure of a dataset folder.

    The function inspects the directory tree rooted at *path* and returns
    a ``DatasetStructure`` describing what it found.  Detection is
    performed in priority order: YOLO, COCO, VOC, DOTA, XLABEL.

    Args:
        path: Root directory of the dataset.

    Returns:
        A populated ``DatasetStructure``.  If the format cannot be
        determined the ``format`` field will be ``DatasetFormat.UNKNOWN``.
    """
    path = osp.abspath(path)
    if not osp.isdir(path):
        logger.warning("Dataset path does not exist or is not a directory: %s", path)
        return DatasetStructure(root_path=path)

    result = DatasetStructure(root_path=path)

    # ----------------------------------------------------------
    # 1.  Try YOLO (highest priority -- data.yaml is definitive)
    # ----------------------------------------------------------
    yolo_result = _try_detect_yolo(path)
    if yolo_result is not None:
        return yolo_result

    # ----------------------------------------------------------
    # 2.  Discover split structure (needed for remaining formats)
    # ----------------------------------------------------------
    split_info = detect_split_structure(path)

    # ----------------------------------------------------------
    # 3.  Try COCO
    # ----------------------------------------------------------
    coco_result = _try_detect_coco(path, split_info)
    if coco_result is not None:
        return coco_result

    # ----------------------------------------------------------
    # 4.  Try VOC
    # ----------------------------------------------------------
    voc_result = _try_detect_voc(path, split_info)
    if voc_result is not None:
        return voc_result

    # ----------------------------------------------------------
    # 5.  Try DOTA
    # ----------------------------------------------------------
    dota_result = _try_detect_dota(path, split_info)
    if dota_result is not None:
        return dota_result

    # ----------------------------------------------------------
    # 6.  Try native X-AnyLabeling (XLABEL)
    # ----------------------------------------------------------
    xlabel_result = _try_detect_xlabel(path, split_info)
    if xlabel_result is not None:
        return xlabel_result

    # ----------------------------------------------------------
    # 7.  Try YOLO without data.yaml (txt labels next to images)
    # ----------------------------------------------------------
    yolo_noyaml_result = _try_detect_yolo_no_yaml(path, split_info)
    if yolo_noyaml_result is not None:
        return yolo_noyaml_result

    # ----------------------------------------------------------
    # 8.  Fallback -- populate images at least
    # ----------------------------------------------------------
    all_images = _scan_images(path)
    if all_images:
        result.splits = {"_all": all_images}
        result.total_images = len(all_images)
    logger.info("Could not determine dataset format for: %s", path)
    return result


# ------------------------------------------------------------------
# Split structure detection
# ------------------------------------------------------------------

def detect_split_structure(path: str) -> Dict[str, str]:
    """Discover train / val / test split directories.

    Handles multiple common layouts:
      - ``images/train/``, ``labels/train/``   (YOLO-style)
      - ``train/images/``, ``train/labels/``    (alternate YOLO)
      - ``train2017/``, ``val2017/``            (COCO-style)
      - ``JPEGImages/``, ``Annotations/``       (VOC-style)
      - Flat folder with no splits

    Args:
        path: Root directory to inspect.

    Returns:
        A dict mapping canonical split name (``"train"``, ``"val"``,
        ``"test"``) to the absolute path of the directory containing
        images for that split.  If no recognisable splits are found
        the dict may contain a single ``"_all"`` key pointing at *path*.
    """
    path = osp.abspath(path)
    if not osp.isdir(path):
        return {}

    entries = _listdir_lower_map(path)
    result: Dict[str, str] = {}

    # --- Pattern 1: images/<split>/ (YOLO canonical) ----------------
    images_dir = entries.get("images")
    if images_dir and osp.isdir(images_dir):
        sub = _listdir_lower_map(images_dir)
        for canon, aliases in _SPLIT_ALIASES.items():
            for alias in aliases:
                if alias in sub and osp.isdir(sub[alias]):
                    result[canon] = sub[alias]
        if result:
            return result

    # --- Pattern 2: <split>/images/ ----------------------------------
    for canon, aliases in _SPLIT_ALIASES.items():
        for alias in aliases:
            if alias in entries and osp.isdir(entries[alias]):
                sub = _listdir_lower_map(entries[alias])
                img_sub = sub.get("images")
                if img_sub and osp.isdir(img_sub):
                    result[canon] = img_sub
    if result:
        return result

    # --- Pattern 3: top-level split dirs (train/, val/, test/) -------
    for canon, aliases in _SPLIT_ALIASES.items():
        for alias in aliases:
            if alias in entries and osp.isdir(entries[alias]):
                result[canon] = entries[alias]
    if result:
        return result

    # --- Pattern 4: VOC JPEGImages/ ---------------------------------
    jpeg_dir = entries.get("jpegimages")
    if jpeg_dir and osp.isdir(jpeg_dir):
        # Check for ImageSets/Main to discover splits
        imagesets_dir = entries.get("imagesets")
        if imagesets_dir and osp.isdir(imagesets_dir):
            main_dir = None
            for name in os.listdir(imagesets_dir):
                if name.lower() == "main":
                    main_dir = osp.join(imagesets_dir, name)
                    break
            if main_dir and osp.isdir(main_dir):
                for fname in os.listdir(main_dir):
                    base, ext = osp.splitext(fname)
                    if ext.lower() == ".txt":
                        canon = _canonical_split_name(base)
                        if canon:
                            result[canon] = jpeg_dir
                if result:
                    return result
        # No ImageSets -- treat as single split
        result["_all"] = jpeg_dir
        return result

    # --- Pattern 5: flat (images directly in root) -------------------
    has_images = any(
        osp.splitext(e)[1].lower() in IMG_FORMATS
        for e in os.listdir(path)
        if osp.isfile(osp.join(path, e))
    )
    if has_images:
        result["_all"] = path

    return result


# ------------------------------------------------------------------
# Format-specific detection helpers
# ------------------------------------------------------------------

def _try_detect_yolo(path: str) -> Optional[DatasetStructure]:
    """Detect YOLO format by looking for data.yaml."""
    yaml_path = None
    for name in os.listdir(path):
        if name.lower() in ("data.yaml", "data.yml", "dataset.yaml", "dataset.yml"):
            candidate = osp.join(path, name)
            if osp.isfile(candidate):
                yaml_path = candidate
                break

    if yaml_path is None:
        return None

    config = _read_yaml_config(yaml_path)
    if config is None:
        return None

    # A YOLO data.yaml must have at least "names" or "nc"
    has_names = "names" in config
    has_nc = "nc" in config
    if not has_names and not has_nc:
        return None

    logger.info("Found YOLO data.yaml: %s", yaml_path)

    result = DatasetStructure(root_path=path)

    # --- classes ---------------------------------------------------
    names = config.get("names", [])
    if isinstance(names, dict):
        # {0: 'person', 1: 'car', ...}
        names = [names[k] for k in sorted(names.keys())]
    if isinstance(names, list):
        result.classes = [str(n) for n in names]
    result.classes_file = yaml_path

    # --- splits from yaml ------------------------------------------
    yaml_dir = osp.dirname(yaml_path)
    for split_key in ("train", "val", "test"):
        raw = config.get(split_key)
        if not raw:
            continue
        if isinstance(raw, list):
            raw = raw[0]
        raw = str(raw)
        # Resolve relative to yaml dir or to path
        candidates = [
            osp.normpath(osp.join(yaml_dir, raw)),
            osp.normpath(osp.join(path, raw)),
        ]
        for cand in candidates:
            if osp.isdir(cand):
                result.splits[split_key] = _scan_images(cand)
                result.annotation_paths[split_key] = _yolo_labels_dir_for(cand)
                break

    # If yaml didn't specify usable paths, fall back to structure
    if not result.splits:
        split_info = detect_split_structure(path)
        for canon, img_dir in split_info.items():
            result.splits[canon] = _scan_images(img_dir)
            result.annotation_paths[canon] = _yolo_labels_dir_for(img_dir)

    # --- determine sub-type from label files -----------------------
    label_samples = _collect_yolo_label_samples(result.annotation_paths, path)
    result.format = _detect_yolo_subtype(label_samples)

    result.total_images = sum(len(v) for v in result.splits.values())
    result.total_annotations = _count_files_in_dirs(
        result.annotation_paths.values(), ".txt"
    )

    return result


def _try_detect_yolo_no_yaml(
    path: str, split_info: Dict[str, str]
) -> Optional[DatasetStructure]:
    """Detect YOLO format when no data.yaml exists but .txt labels do."""
    label_dirs: Dict[str, str] = {}
    image_splits: Dict[str, List[str]] = {}

    for canon, img_dir in split_info.items():
        labels_dir = _yolo_labels_dir_for(img_dir)
        txt_files = _sample_annotation_files(labels_dir, ".txt", n=3)
        if txt_files:
            label_dirs[canon] = labels_dir
            image_splits[canon] = _scan_images(img_dir)

    # Also check the flat case
    if not label_dirs:
        txt_files = _sample_annotation_files(path, ".txt", n=3)
        if txt_files and _scan_images(path):
            # Verify at least one txt looks like YOLO
            if _looks_like_yolo_txt(txt_files):
                label_dirs["_all"] = path
                image_splits["_all"] = _scan_images(path)

    if not label_dirs:
        return None

    # Validate that the txt files look like YOLO, not DOTA
    all_samples: List[str] = []
    for d in label_dirs.values():
        all_samples.extend(_sample_annotation_files(d, ".txt", n=3))
    if not _looks_like_yolo_txt(all_samples):
        return None

    result = DatasetStructure(root_path=path)
    result.splits = image_splits
    result.annotation_paths = label_dirs

    # Try to find classes.txt
    classes_file = _find_classes_file(path)
    if classes_file:
        result.classes = _read_classes_txt(classes_file)
        result.classes_file = classes_file

    result.format = _detect_yolo_subtype(all_samples)
    result.total_images = sum(len(v) for v in result.splits.values())
    result.total_annotations = _count_files_in_dirs(
        result.annotation_paths.values(), ".txt"
    )
    return result


def _try_detect_coco(
    path: str, split_info: Dict[str, str]
) -> Optional[DatasetStructure]:
    """Detect COCO JSON format."""
    coco_jsons = _find_coco_jsons(path)
    if not coco_jsons:
        return None

    result = DatasetStructure(root_path=path)
    result.coco_json_paths = {}

    # Determine sub-type from the first available JSON
    fmt = DatasetFormat.COCO_DETECT
    first_data = None
    for split, json_path in coco_jsons.items():
        result.coco_json_paths[split] = json_path
        result.annotation_paths[split] = json_path
        if first_data is None:
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    first_data = json.load(f)
                fmt = _detect_coco_subtype(first_data)
            except Exception:
                logger.debug("Failed to parse COCO JSON: %s", json_path)

    result.format = fmt

    # Populate splits with image paths
    if split_info:
        for canon, img_dir in split_info.items():
            images = _scan_images(img_dir)
            if images:
                result.splits[canon] = images
    # If split_info didn't yield images, try to infer from COCO JSON
    if not result.splits and first_data:
        images_data = first_data.get("images", [])
        # Try common image directories
        for candidate_dir_name in ["images", "train2017", "val2017", ""]:
            candidate = osp.join(path, candidate_dir_name) if candidate_dir_name else path
            if osp.isdir(candidate) and _scan_images(candidate):
                result.splits["_all"] = _scan_images(candidate)
                break

    # Extract categories
    if first_data:
        categories = first_data.get("categories", [])
        result.classes = [
            cat["name"] for cat in sorted(categories, key=lambda c: c.get("id", 0))
            if "name" in cat
        ]
        result.total_annotations = len(first_data.get("annotations", []))

    result.total_images = sum(len(v) for v in result.splits.values())
    if not result.total_images and first_data:
        result.total_images = len(first_data.get("images", []))

    return result


def _try_detect_voc(
    path: str, split_info: Dict[str, str]
) -> Optional[DatasetStructure]:
    """Detect Pascal VOC XML format."""
    entries = _listdir_lower_map(path)

    # Standard VOC: Annotations/ + JPEGImages/
    ann_dir = entries.get("annotations")
    jpeg_dir = entries.get("jpegimages")

    if ann_dir and osp.isdir(ann_dir):
        xml_samples = _sample_annotation_files(ann_dir, ".xml", n=5)
        if xml_samples and _looks_like_voc_xml(xml_samples):
            return _build_voc_result(path, ann_dir, split_info, xml_samples)

    # Check split dirs for .xml files
    for canon, img_dir in split_info.items():
        # Annotations might be in a sibling directory
        parent = osp.dirname(img_dir)
        for ann_name in ["Annotations", "annotations", "labels", "xmls"]:
            candidate = osp.join(parent, ann_name)
            if osp.isdir(candidate):
                xml_samples = _sample_annotation_files(candidate, ".xml", n=5)
                if xml_samples and _looks_like_voc_xml(xml_samples):
                    return _build_voc_result(
                        path, candidate, split_info, xml_samples
                    )

    # Flat: .xml files alongside images
    xml_samples = _sample_annotation_files(path, ".xml", n=5)
    if xml_samples and _looks_like_voc_xml(xml_samples):
        return _build_voc_result(path, path, split_info, xml_samples)

    return None


def _try_detect_dota(
    path: str, split_info: Dict[str, str]
) -> Optional[DatasetStructure]:
    """Detect DOTA format (.txt with x1 y1 ... x4 y4 category difficult)."""
    dirs_to_check: List[str] = []

    # Check label directories in split_info
    for canon, img_dir in split_info.items():
        labels_dir = _yolo_labels_dir_for(img_dir)
        if osp.isdir(labels_dir):
            dirs_to_check.append(labels_dir)

    # Also check labelTxt directories (common in DOTA)
    entries = _listdir_lower_map(path)
    for key in ("labeltxt", "labelstxt", "labels", "annotations"):
        if key in entries and osp.isdir(entries[key]):
            dirs_to_check.append(entries[key])

    # Flat
    dirs_to_check.append(path)

    for d in dirs_to_check:
        txt_samples = _sample_annotation_files(d, ".txt", n=5)
        if txt_samples and _looks_like_dota_txt(txt_samples):
            result = DatasetStructure(
                root_path=path, format=DatasetFormat.DOTA
            )
            if split_info:
                for canon, img_dir in split_info.items():
                    result.splits[canon] = _scan_images(img_dir)
                    result.annotation_paths[canon] = d
            else:
                result.splits["_all"] = _scan_images(path)
                result.annotation_paths["_all"] = d

            classes_file = _find_classes_file(path)
            if classes_file:
                result.classes = _read_classes_txt(classes_file)
                result.classes_file = classes_file
            else:
                result.classes = _extract_dota_classes(d)

            result.total_images = sum(len(v) for v in result.splits.values())
            result.total_annotations = _count_files_in_dirs([d], ".txt")
            return result

    return None


def _try_detect_xlabel(
    path: str, split_info: Dict[str, str]
) -> Optional[DatasetStructure]:
    """Detect native X-AnyLabeling JSON format (has ``"shapes"`` key)."""
    dirs_to_check = list(split_info.values()) if split_info else [path]

    for d in dirs_to_check:
        json_samples = _sample_annotation_files(d, ".json", n=5)
        if json_samples and _looks_like_xlabel_json(json_samples):
            result = DatasetStructure(
                root_path=path, format=DatasetFormat.XLABEL
            )
            if split_info:
                for canon, img_dir in split_info.items():
                    result.splits[canon] = _scan_images(img_dir)
                    result.annotation_paths[canon] = img_dir
            else:
                result.splits["_all"] = _scan_images(path)
                result.annotation_paths["_all"] = path

            # Extract class names from a few JSON files
            result.classes = _extract_xlabel_classes(d)

            result.total_images = sum(len(v) for v in result.splits.values())
            result.total_annotations = _count_files_in_dir(d, ".json")
            return result

    return None


# ------------------------------------------------------------------
# Helper: image scanning
# ------------------------------------------------------------------

def _scan_images(directory: str) -> List[str]:
    """Find all image files in *directory* (non-recursive).

    Args:
        directory: Absolute path to scan.

    Returns:
        Sorted list of absolute image file paths.
    """
    if not osp.isdir(directory):
        return []
    result = []
    try:
        for name in os.listdir(directory):
            if osp.splitext(name)[1].lower() in IMG_FORMATS:
                full = osp.join(directory, name)
                if osp.isfile(full):
                    result.append(full)
    except OSError as exc:
        logger.debug("Error scanning images in %s: %s", directory, exc)
    result.sort()
    return result


# ------------------------------------------------------------------
# Helper: YAML parsing
# ------------------------------------------------------------------

def _read_yaml_config(path: str) -> Optional[dict]:
    """Safely parse a YAML file.

    Args:
        path: Absolute path to the YAML file.

    Returns:
        Parsed dict, or ``None`` on failure.
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        if isinstance(data, dict):
            return data
    except Exception as exc:
        logger.debug("Failed to parse YAML %s: %s", path, exc)
    return None


# ------------------------------------------------------------------
# Helper: annotation file sampling
# ------------------------------------------------------------------

def _sample_annotation_files(
    directory: str, ext: str, n: int = 5
) -> List[str]:
    """Return up to *n* annotation file paths from *directory*.

    Files are chosen pseudo-randomly so that the sample is
    representative even for large datasets.

    Args:
        directory: Directory to search.
        ext: File extension including the dot (e.g. ``".txt"``).
        n: Maximum number of files to return.

    Returns:
        List of absolute file paths (may be shorter than *n*).
    """
    if not osp.isdir(directory):
        return []
    try:
        candidates = [
            osp.join(directory, f)
            for f in os.listdir(directory)
            if f.lower().endswith(ext) and osp.isfile(osp.join(directory, f))
        ]
    except OSError:
        return []
    if len(candidates) <= n:
        return candidates
    # Deterministic seed so repeated calls give the same sample
    rng = random.Random(42)
    return rng.sample(candidates, n)


# ------------------------------------------------------------------
# Helper: YOLO specifics
# ------------------------------------------------------------------

def _parse_yolo_line(line: str) -> Tuple[int, List[float]]:
    """Parse a single YOLO annotation line.

    Args:
        line: A whitespace-separated line from a YOLO label file.

    Returns:
        Tuple of (class_id, list_of_float_values).

    Raises:
        ValueError: If the line cannot be parsed.
    """
    parts = line.strip().split()
    if len(parts) < 2:
        raise ValueError(f"Too few values in YOLO line: {line!r}")
    class_id = int(parts[0])
    values = [float(v) for v in parts[1:]]
    return class_id, values


def _detect_yolo_subtype(label_files: List[str]) -> DatasetFormat:
    """Determine the YOLO sub-type by sampling label file contents.

    Args:
        label_files: Paths to ``.txt`` label files.

    Returns:
        One of ``YOLO_DETECT``, ``YOLO_SEG``, ``YOLO_OBB``,
        ``YOLO_POSE``, or ``YOLO_DETECT`` as default.
    """
    value_counts: List[int] = []

    for fpath in label_files:
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                for raw_line in f:
                    raw_line = raw_line.strip()
                    if not raw_line:
                        continue
                    try:
                        _, values = _parse_yolo_line(raw_line)
                        value_counts.append(len(values))
                    except ValueError:
                        continue
        except OSError:
            continue

    if not value_counts:
        return DatasetFormat.YOLO_DETECT

    # Use the most common value count for classification
    from collections import Counter

    counter = Counter(value_counts)
    most_common_count = counter.most_common(1)[0][0]

    # OBB: exactly 8 coordinate values (4 corner points)
    if most_common_count == 8:
        return DatasetFormat.YOLO_OBB

    # Detect: exactly 4 values (cx, cy, w, h)
    if most_common_count == 4:
        return DatasetFormat.YOLO_DETECT

    # Pose: 4 + 3*n values where n >= 1  (bbox + keypoints with visibility)
    # i.e. 7, 10, 13, 16, ... or 4 + 2*n (without visibility): 6, 8, 10, ...
    # The canonical Ultralytics pose format uses 4 + 3*n
    if most_common_count > 4 and (most_common_count - 4) % 3 == 0:
        return DatasetFormat.YOLO_POSE

    # Seg: variable number of values > 4, even number of coordinates
    # (class_id followed by pairs of x, y polygon points)
    if most_common_count > 4:
        # Check whether values > 4 and the coord count (values) is even
        # indicating polygon x,y pairs
        coords_count = most_common_count  # values after class_id
        if coords_count % 2 == 0 and coords_count >= 6:
            return DatasetFormat.YOLO_SEG
        # Could still be pose with visibility (4 + 2*n without vis flag)
        if (coords_count - 4) % 2 == 0 and coords_count > 4:
            return DatasetFormat.YOLO_POSE
        return DatasetFormat.YOLO_SEG

    return DatasetFormat.YOLO_DETECT


def _yolo_labels_dir_for(images_dir: str) -> str:
    """Infer the labels directory corresponding to an images directory.

    Common patterns:
      - ``images/train`` -> ``labels/train``
      - ``train/images`` -> ``train/labels``

    Args:
        images_dir: Path to the images directory.

    Returns:
        Best-guess path for the matching labels directory.
        May not actually exist on disk.
    """
    images_dir = osp.normpath(images_dir)

    # Pattern: .../images/<split> -> .../labels/<split>
    parent = osp.dirname(images_dir)
    basename = osp.basename(images_dir)
    grandparent = osp.dirname(parent)
    parent_name = osp.basename(parent)

    if parent_name.lower() == "images":
        candidate = osp.join(grandparent, "labels", basename)
        if osp.isdir(candidate):
            return candidate

    # Pattern: .../<split>/images -> .../<split>/labels
    if basename.lower() == "images":
        candidate = osp.join(parent, "labels")
        if osp.isdir(candidate):
            return candidate

    # Fallback: sibling "labels" directory at same level
    candidate = osp.join(parent, "labels")
    if osp.isdir(candidate):
        return candidate

    # Last resort: same directory (flat structure)
    return images_dir


def _collect_yolo_label_samples(
    annotation_paths: Dict[str, str], root: str
) -> List[str]:
    """Gather sample .txt label files from known annotation paths."""
    samples: List[str] = []
    for d in annotation_paths.values():
        samples.extend(_sample_annotation_files(d, ".txt", n=3))
    if not samples:
        # Try root-level labels/ directory
        labels_dir = osp.join(root, "labels")
        if osp.isdir(labels_dir):
            samples.extend(_sample_annotation_files(labels_dir, ".txt", n=5))
    return samples


def _looks_like_yolo_txt(txt_files: List[str]) -> bool:
    """Return True if the sampled .txt files look like YOLO annotations."""
    for fpath in txt_files:
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                for raw_line in f:
                    raw_line = raw_line.strip()
                    if not raw_line:
                        continue
                    parts = raw_line.split()
                    if len(parts) < 5:
                        return False
                    try:
                        int(parts[0])
                        for p in parts[1:]:
                            float(p)
                    except ValueError:
                        return False
                    return True
        except OSError:
            continue
    return False


# ------------------------------------------------------------------
# Helper: COCO specifics
# ------------------------------------------------------------------

def _find_coco_jsons(path: str) -> Dict[str, str]:
    """Locate COCO-style JSON annotation files.

    Searches for:
      - ``annotations/instances_train*.json``, ``annotations/instances_val*.json``
      - ``_annotations.coco.json`` (Roboflow format)
      - Any top-level JSON with ``"images"`` and ``"annotations"`` keys

    Returns:
        Dict mapping split name to JSON file path.
    """
    result: Dict[str, str] = {}

    # 1. Standard annotations/ directory
    entries = _listdir_lower_map(path)
    ann_dir = entries.get("annotations")
    if ann_dir and osp.isdir(ann_dir):
        try:
            for fname in os.listdir(ann_dir):
                if not fname.lower().endswith(".json"):
                    continue
                full = osp.join(ann_dir, fname)
                lower = fname.lower()
                # instances_train2017.json, instances_val2017.json, etc.
                if lower.startswith("instances_"):
                    split = _canonical_split_name(lower.replace("instances_", "").replace(".json", ""))
                    if split:
                        result[split] = full
                        continue
                # person_keypoints_train2017.json, etc.
                if "keypoints" in lower or "captions" in lower:
                    split = _extract_split_from_filename(lower)
                    if split and split not in result:
                        result[split] = full
                        continue
                # Generic: check if it's valid COCO
                if _is_coco_json(full):
                    split = _extract_split_from_filename(lower) or "_all"
                    if split not in result:
                        result[split] = full
        except OSError:
            pass

    if result:
        return result

    # 2. Roboflow style: <split>/_annotations.coco.json
    for canon, aliases in _SPLIT_ALIASES.items():
        for alias in aliases:
            if alias in entries and osp.isdir(entries[alias]):
                candidate = osp.join(entries[alias], "_annotations.coco.json")
                if osp.isfile(candidate):
                    result[canon] = candidate
    if result:
        return result

    # 3. Any JSON in root that looks like COCO
    try:
        for fname in os.listdir(path):
            if not fname.lower().endswith(".json"):
                continue
            full = osp.join(path, fname)
            if osp.isfile(full) and _is_coco_json(full):
                split = _extract_split_from_filename(fname.lower()) or "_all"
                if split not in result:
                    result[split] = full
    except OSError:
        pass

    return result


def _is_coco_json(filepath: str) -> bool:
    """Quick check whether a JSON file has COCO structure."""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            # Read only the beginning to check for keys without
            # parsing the entire (potentially huge) file.
            head = f.read(4096)
        return '"images"' in head and '"annotations"' in head
    except (OSError, UnicodeDecodeError):
        return False


def _detect_coco_subtype(coco_json: dict) -> DatasetFormat:
    """Determine the COCO sub-type from parsed JSON data.

    Args:
        coco_json: Parsed COCO JSON dict.

    Returns:
        ``COCO_DETECT``, ``COCO_SEG``, or ``COCO_POSE``.
    """
    annotations = coco_json.get("annotations", [])
    if not annotations:
        return DatasetFormat.COCO_DETECT

    # Sample a few annotations
    sample = annotations[:min(10, len(annotations))]

    has_keypoints = False
    has_segmentation = False

    for ann in sample:
        kps = ann.get("keypoints")
        if kps and isinstance(kps, list) and len(kps) > 0:
            # Check that keypoints are not all zeros
            if any(v != 0 for v in kps):
                has_keypoints = True

        seg = ann.get("segmentation")
        if seg:
            if isinstance(seg, list) and len(seg) > 0:
                has_segmentation = True
            elif isinstance(seg, dict) and seg.get("counts"):
                # RLE format
                has_segmentation = True

    if has_keypoints:
        return DatasetFormat.COCO_POSE
    if has_segmentation:
        return DatasetFormat.COCO_SEG
    return DatasetFormat.COCO_DETECT


# ------------------------------------------------------------------
# Helper: VOC specifics
# ------------------------------------------------------------------

def _looks_like_voc_xml(xml_files: List[str]) -> bool:
    """Return True if sampled XML files look like Pascal VOC annotations."""
    for fpath in xml_files:
        try:
            tree = ET.parse(fpath)
            root = tree.getroot()
            if root.tag != "annotation":
                return False
            # Must have at least a filename or folder element
            if root.find("filename") is not None or root.find("folder") is not None:
                return True
        except (ET.ParseError, OSError):
            continue
    return False


def _build_voc_result(
    path: str,
    ann_dir: str,
    split_info: Dict[str, str],
    xml_samples: List[str],
) -> DatasetStructure:
    """Build a DatasetStructure for a VOC dataset."""
    result = DatasetStructure(root_path=path)

    # Determine detect vs seg
    has_seg = False
    classes_set: List[str] = []
    for fpath in xml_samples:
        try:
            tree = ET.parse(fpath)
            root = tree.getroot()
            seg_elem = root.find("segmented")
            if seg_elem is not None and seg_elem.text and seg_elem.text.strip() == "1":
                has_seg = True
            for obj in root.findall("object"):
                name_elem = obj.find("name")
                if name_elem is not None and name_elem.text:
                    cls = name_elem.text.strip()
                    if cls not in classes_set:
                        classes_set.append(cls)
        except (ET.ParseError, OSError):
            continue

    result.format = DatasetFormat.VOC_SEG if has_seg else DatasetFormat.VOC_DETECT

    if split_info:
        for canon, img_dir in split_info.items():
            result.splits[canon] = _scan_images(img_dir)
            result.annotation_paths[canon] = ann_dir
    else:
        entries = _listdir_lower_map(path)
        jpeg_dir = entries.get("jpegimages", path)
        result.splits["_all"] = _scan_images(jpeg_dir)
        result.annotation_paths["_all"] = ann_dir

    result.classes = classes_set

    # Try to load full class list from a dedicated file
    classes_file = _find_classes_file(path)
    if classes_file:
        full_classes = _read_classes_txt(classes_file)
        if full_classes:
            result.classes = full_classes
            result.classes_file = classes_file

    result.total_images = sum(len(v) for v in result.splits.values())
    result.total_annotations = _count_files_in_dir(ann_dir, ".xml")
    return result


# ------------------------------------------------------------------
# Helper: DOTA specifics
# ------------------------------------------------------------------

def _looks_like_dota_txt(txt_files: List[str]) -> bool:
    """Return True if sampled .txt files match DOTA format.

    DOTA lines: ``x1 y1 x2 y2 x3 y3 x4 y4 category difficult``
    (10 space-separated tokens, last two are string and int).
    """
    for fpath in txt_files:
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                for raw_line in f:
                    raw_line = raw_line.strip()
                    if not raw_line:
                        continue
                    # Skip DOTA header lines (e.g. "imagesource:..." or "gsd:...")
                    if ":" in raw_line:
                        continue
                    parts = raw_line.split()
                    if len(parts) < 9:
                        return False
                    # First 8 should be numeric (coordinates)
                    try:
                        for p in parts[:8]:
                            float(p)
                    except ValueError:
                        return False
                    # 9th is category name (string), 10th is difficult (int)
                    # The category is non-numeric in most cases
                    category = parts[8]
                    try:
                        # If the 9th field is purely numeric, this is probably
                        # not DOTA (could be YOLO OBB)
                        float(category)
                        return False
                    except ValueError:
                        pass
                    return True
        except OSError:
            continue
    return False


def _extract_dota_classes(label_dir: str) -> List[str]:
    """Extract unique class names from DOTA label files."""
    classes: List[str] = []
    try:
        for fname in os.listdir(label_dir):
            if not fname.lower().endswith(".txt"):
                continue
            fpath = osp.join(label_dir, fname)
            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    for raw_line in f:
                        raw_line = raw_line.strip()
                        if not raw_line or ":" in raw_line:
                            continue
                        parts = raw_line.split()
                        if len(parts) >= 9:
                            cls = parts[8]
                            if cls not in classes:
                                classes.append(cls)
            except OSError:
                continue
            # Stop after scanning a reasonable number of files
            if len(classes) > 50:
                break
    except OSError:
        pass
    return classes


# ------------------------------------------------------------------
# Helper: X-AnyLabeling (XLABEL) specifics
# ------------------------------------------------------------------

def _looks_like_xlabel_json(json_files: List[str]) -> bool:
    """Return True if sampled JSON files look like X-AnyLabeling format."""
    for fpath in json_files:
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                head = f.read(2048)
            if '"shapes"' in head:
                return True
        except (OSError, UnicodeDecodeError):
            continue
    return False


def _extract_xlabel_classes(directory: str) -> List[str]:
    """Extract class names from X-AnyLabeling JSON files."""
    classes: List[str] = []
    json_files = _sample_annotation_files(directory, ".json", n=10)
    for fpath in json_files:
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, dict):
                continue
            for shape in data.get("shapes", []):
                label = shape.get("label", "")
                if label and label not in classes:
                    classes.append(label)
        except (OSError, json.JSONDecodeError, KeyError):
            continue
    return classes


# ------------------------------------------------------------------
# Helper: classes file discovery
# ------------------------------------------------------------------

def _find_classes_file(root: str) -> Optional[str]:
    """Search for a classes definition file in the dataset root.

    Looks for: ``classes.txt``, ``labels.txt``, ``obj.names``,
    ``names.txt``, ``classes.names``.

    Returns:
        Absolute path if found, else ``None``.
    """
    candidates = [
        "classes.txt",
        "labels.txt",
        "obj.names",
        "names.txt",
        "classes.names",
    ]
    for name in candidates:
        full = osp.join(root, name)
        if osp.isfile(full):
            return full
    return None


def _read_classes_txt(filepath: str) -> List[str]:
    """Read a newline-delimited classes file.

    Returns:
        List of class names (empty lines stripped).
    """
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip()]
    except OSError as exc:
        logger.debug("Failed to read classes file %s: %s", filepath, exc)
        return []


# ------------------------------------------------------------------
# Generic path / file utilities
# ------------------------------------------------------------------

def _listdir_lower_map(directory: str) -> Dict[str, str]:
    """Map lower-cased entry names to their full paths.

    This enables case-insensitive lookups of directory entries
    which is important on case-sensitive filesystems where the
    actual casing may vary between datasets.

    Args:
        directory: Directory to list.

    Returns:
        Dict mapping ``entry_name.lower()`` to absolute path.
    """
    result: Dict[str, str] = {}
    try:
        for name in os.listdir(directory):
            result[name.lower()] = osp.join(directory, name)
    except OSError:
        pass
    return result


def _canonical_split_name(raw: str) -> Optional[str]:
    """Map a raw string to a canonical split name.

    Args:
        raw: A string like ``"train2017"`` or ``"valid"``.

    Returns:
        ``"train"``, ``"val"``, or ``"test"``; or ``None`` if
        the string doesn't match any known alias.
    """
    raw = raw.lower().strip()
    for canon, aliases in _SPLIT_ALIASES.items():
        for alias in aliases:
            if raw == alias or raw.startswith(alias):
                return canon
    return None


def _extract_split_from_filename(filename: str) -> Optional[str]:
    """Try to extract a canonical split name from a filename."""
    lower = filename.lower()
    for canon, aliases in _SPLIT_ALIASES.items():
        for alias in aliases:
            if alias in lower:
                return canon
    return None


def _count_files_in_dir(directory: str, ext: str) -> int:
    """Count files with a given extension in a single directory."""
    if not osp.isdir(directory):
        return 0
    count = 0
    try:
        for name in os.listdir(directory):
            if name.lower().endswith(ext) and osp.isfile(osp.join(directory, name)):
                count += 1
    except OSError:
        pass
    return count


def _count_files_in_dirs(directories, ext: str) -> int:
    """Count files with a given extension across multiple directories."""
    seen: set = set()
    total = 0
    for d in directories:
        d = osp.normpath(d)
        if d in seen:
            continue
        seen.add(d)
        total += _count_files_in_dir(d, ext)
    return total
