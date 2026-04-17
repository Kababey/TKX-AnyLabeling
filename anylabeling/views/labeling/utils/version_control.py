"""Version control module for labeled datasets.

Provides lightweight snapshot-based version control so users can create
named snapshots of their annotations, browse history, compare versions,
and restore previous versions.  This is a pure logic module with no UI
dependencies.
"""

import json
import math
import os
import os.path as osp
import shutil
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from anylabeling.views.labeling.logger import logger


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class VersionInfo:
    """Immutable descriptor for a single version snapshot."""

    version_id: str
    name: str
    description: str
    timestamp: str
    stats: Dict
    size_bytes: int


@dataclass
class ImageDiff:
    """Per-image difference record between two versions."""

    image_name: str
    status: str  # "added", "removed", "modified", "unchanged"
    added_shapes: int = 0
    removed_shapes: int = 0
    modified_shapes: int = 0
    details: List[Dict] = field(default_factory=list)


@dataclass
class VersionDiff:
    """Aggregate diff between two version snapshots."""

    version_a: str
    version_b: str
    image_diffs: List[ImageDiff] = field(default_factory=list)
    summary: Dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

_POINT_TOLERANCE = 2.0  # pixels


def _load_annotation(json_path: str) -> Optional[Dict]:
    """Safely load an XLABEL JSON annotation file.

    Returns the parsed dict on success, or ``None`` on any error.
    """
    try:
        with open(json_path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        logger.warning(
            "Failed to load annotation file '%s': %s", json_path, exc
        )
        return None


def _points_distance(pts_a: List[List[float]], pts_b: List[List[float]]) -> float:
    """Compute the average Euclidean distance between two point lists.

    If the lists differ in length the result is ``float('inf')``.
    """
    if len(pts_a) != len(pts_b):
        return float("inf")
    if not pts_a:
        return 0.0
    total = 0.0
    for (ax, ay), (bx, by) in zip(pts_a, pts_b):
        total += math.hypot(ax - bx, ay - by)
    return total / len(pts_a)


def _shapes_match(shape_a: Dict, shape_b: Dict) -> bool:
    """Return ``True`` if two shapes are considered the same annotation.

    Two shapes match when they share the same *label*, *shape_type*, and
    their point lists are within ``_POINT_TOLERANCE`` pixels on average.
    """
    if shape_a.get("label") != shape_b.get("label"):
        return False
    if shape_a.get("shape_type") != shape_b.get("shape_type"):
        return False
    pts_a = shape_a.get("points", [])
    pts_b = shape_b.get("points", [])
    return _points_distance(pts_a, pts_b) <= _POINT_TOLERANCE


def _compare_shape_lists(
    shapes_a: List[Dict], shapes_b: List[Dict]
) -> Tuple[int, int, int, List[Dict]]:
    """Compare two lists of shape dicts.

    Uses a greedy matching algorithm: for each shape in *shapes_a*, find the
    best match in *shapes_b* by label + shape_type + points proximity.

    Returns:
        A 4-tuple of ``(added, removed, modified, details)`` where *details*
        is a list of per-shape change records.
    """
    used_b: List[bool] = [False] * len(shapes_b)
    matched_a: List[bool] = [False] * len(shapes_a)
    details: List[Dict] = []

    # --- Pass 1: exact matches (within tolerance) --------------------------
    for i, sa in enumerate(shapes_a):
        best_j: Optional[int] = None
        best_dist = float("inf")
        for j, sb in enumerate(shapes_b):
            if used_b[j]:
                continue
            if sa.get("label") != sb.get("label"):
                continue
            if sa.get("shape_type") != sb.get("shape_type"):
                continue
            dist = _points_distance(
                sa.get("points", []), sb.get("points", [])
            )
            if dist < best_dist:
                best_dist = dist
                best_j = j
        if best_j is not None and best_dist <= _POINT_TOLERANCE:
            used_b[best_j] = True
            matched_a[i] = True
            # Exact match -> unchanged, no detail entry needed.

    # --- Pass 2: modified shapes (same label+type, points differ) ----------
    modified = 0
    for i, sa in enumerate(shapes_a):
        if matched_a[i]:
            continue
        best_j: Optional[int] = None
        best_dist = float("inf")
        for j, sb in enumerate(shapes_b):
            if used_b[j]:
                continue
            if sa.get("label") != sb.get("label"):
                continue
            if sa.get("shape_type") != sb.get("shape_type"):
                continue
            dist = _points_distance(
                sa.get("points", []), sb.get("points", [])
            )
            if dist < best_dist:
                best_dist = dist
                best_j = j
        if best_j is not None and best_dist < float("inf"):
            used_b[best_j] = True
            matched_a[i] = True
            modified += 1
            details.append(
                {
                    "label": sa.get("label", ""),
                    "shape_type": sa.get("shape_type", ""),
                    "change_type": "modified",
                }
            )

    # --- Pass 3: remaining unmatched in A -> removed -----------------------
    removed = 0
    for i, sa in enumerate(shapes_a):
        if matched_a[i]:
            continue
        removed += 1
        details.append(
            {
                "label": sa.get("label", ""),
                "shape_type": sa.get("shape_type", ""),
                "change_type": "removed",
            }
        )

    # --- Pass 4: remaining unmatched in B -> added -------------------------
    added = 0
    for j, sb in enumerate(shapes_b):
        if used_b[j]:
            continue
        added += 1
        details.append(
            {
                "label": sb.get("label", ""),
                "shape_type": sb.get("shape_type", ""),
                "change_type": "added",
            }
        )

    return added, removed, modified, details


# ---------------------------------------------------------------------------
# VersionManager
# ---------------------------------------------------------------------------


class VersionManager:
    """Manage lightweight annotation snapshots for a labeling project.

    Parameters
    ----------
    project_dir:
        Root directory of the labeling project (where images reside).
    output_dir:
        Optional override for where the ``.xanylabeling_versions/``
        directory is created.  Defaults to *project_dir*.
    """

    _VERSIONS_DIR_NAME = ".xanylabeling_versions"

    def __init__(self, project_dir: str, output_dir: Optional[str] = None):
        self._project_dir = project_dir
        base = output_dir if output_dir else project_dir
        self._versions_root = osp.join(base, self._VERSIONS_DIR_NAME)
        try:
            os.makedirs(self._versions_root, exist_ok=True)
        except OSError as exc:
            logger.error(
                "Failed to create versions directory '%s': %s",
                self._versions_root,
                exc,
            )

    # -- public helpers -----------------------------------------------------

    def get_versions_root(self) -> str:
        """Return the absolute path to the ``.xanylabeling_versions/`` directory."""
        return self._versions_root

    def get_version_dir(self, version_id: str) -> str:
        """Return the absolute path to a specific version's directory."""
        return osp.join(self._versions_root, version_id)

    # -- CRUD ---------------------------------------------------------------

    def create_version(
        self,
        name: str,
        description: str,
        image_list: List[str],
        label_dir: str,
    ) -> VersionInfo:
        """Create a new version snapshot.

        Parameters
        ----------
        name:
            Human-readable version name.
        description:
            Free-form description of this snapshot.
        image_list:
            List of image filenames (e.g. ``["img001.jpg", "img002.png"]``).
            Their corresponding ``.json`` label files will be looked up in
            *label_dir*.
        label_dir:
            Directory containing the XLABEL JSON label files.

        Returns
        -------
        VersionInfo
            Metadata of the newly created version.
        """
        now = datetime.now()
        suffix = uuid.uuid4().hex[:6]
        version_id = now.strftime(f"v_%Y%m%d_%H%M%S_{suffix}")

        version_dir = osp.join(self._versions_root, version_id)
        annotations_dir = osp.join(version_dir, "annotations")
        try:
            os.makedirs(annotations_dir, exist_ok=True)
        except OSError as exc:
            logger.error(
                "Failed to create version directory '%s': %s",
                annotations_dir,
                exc,
            )
            raise

        # Copy matching label files
        copied = 0
        for img_name in image_list:
            base_name = osp.splitext(img_name)[0]
            json_name = base_name + ".json"
            src = osp.join(label_dir, json_name)
            if osp.isfile(src):
                try:
                    shutil.copy2(src, osp.join(annotations_dir, json_name))
                    copied += 1
                except OSError as exc:
                    logger.warning(
                        "Failed to copy '%s': %s", src, exc
                    )

        logger.info(
            "Version '%s': copied %d annotation files.", version_id, copied
        )

        stats = self._compute_annotation_stats(annotations_dir, image_list)
        size_bytes = self._compute_dir_size(version_dir)

        info = VersionInfo(
            version_id=version_id,
            name=name,
            description=description,
            timestamp=now.isoformat(timespec="seconds"),
            stats=stats,
            size_bytes=size_bytes,
        )

        metadata_path = osp.join(version_dir, "metadata.json")
        try:
            with open(metadata_path, "w", encoding="utf-8") as fh:
                json.dump(asdict(info), fh, indent=2, ensure_ascii=False)
        except OSError as exc:
            logger.error(
                "Failed to write metadata for version '%s': %s",
                version_id,
                exc,
            )
            raise

        logger.info("Created version '%s' (%s).", version_id, name)
        return info

    def list_versions(self) -> List[VersionInfo]:
        """Return all versions sorted by timestamp (newest first).

        Versions whose ``metadata.json`` cannot be read are silently
        skipped.
        """
        versions: List[VersionInfo] = []
        try:
            entries = os.listdir(self._versions_root)
        except OSError as exc:
            logger.error(
                "Failed to list versions directory '%s': %s",
                self._versions_root,
                exc,
            )
            return versions

        for entry in entries:
            entry_path = osp.join(self._versions_root, entry)
            if not osp.isdir(entry_path):
                continue
            meta_path = osp.join(entry_path, "metadata.json")
            if not osp.isfile(meta_path):
                continue
            info = self._read_metadata(meta_path)
            if info is not None:
                versions.append(info)

        versions.sort(key=lambda v: v.timestamp, reverse=True)
        return versions

    def get_version(self, version_id: str) -> Optional[VersionInfo]:
        """Read and return metadata for a single version, or ``None``."""
        meta_path = osp.join(
            self._versions_root, version_id, "metadata.json"
        )
        if not osp.isfile(meta_path):
            logger.warning(
                "Version '%s' not found (no metadata.json).", version_id
            )
            return None
        return self._read_metadata(meta_path)

    def delete_version(self, version_id: str) -> bool:
        """Delete a version snapshot and all its files.

        Returns ``True`` on success, ``False`` otherwise.
        """
        version_dir = osp.join(self._versions_root, version_id)
        if not osp.isdir(version_dir):
            logger.warning(
                "Cannot delete version '%s': directory does not exist.",
                version_id,
            )
            return False
        try:
            shutil.rmtree(version_dir)
            logger.info("Deleted version '%s'.", version_id)
            return True
        except OSError as exc:
            logger.error(
                "Failed to delete version '%s': %s", version_id, exc
            )
            return False

    def restore_version(self, version_id: str, target_dir: str) -> int:
        """Restore annotation files from a version snapshot.

        Copies all ``.json`` files from the version's ``annotations/``
        directory into *target_dir*, overwriting existing files.

        Returns the number of files restored.
        """
        annotations_dir = osp.join(
            self._versions_root, version_id, "annotations"
        )
        if not osp.isdir(annotations_dir):
            logger.error(
                "Cannot restore version '%s': annotations directory missing.",
                version_id,
            )
            return 0

        try:
            os.makedirs(target_dir, exist_ok=True)
        except OSError as exc:
            logger.error(
                "Failed to create target directory '%s': %s",
                target_dir,
                exc,
            )
            return 0

        count = 0
        try:
            filenames = os.listdir(annotations_dir)
        except OSError as exc:
            logger.error(
                "Failed to list annotations in version '%s': %s",
                version_id,
                exc,
            )
            return 0

        for fname in filenames:
            if not fname.lower().endswith(".json"):
                continue
            src = osp.join(annotations_dir, fname)
            dst = osp.join(target_dir, fname)
            try:
                shutil.copy2(src, dst)
                count += 1
            except OSError as exc:
                logger.warning(
                    "Failed to restore '%s' -> '%s': %s", src, dst, exc
                )

        logger.info(
            "Restored %d annotation files from version '%s' to '%s'.",
            count,
            version_id,
            target_dir,
        )
        return count

    def compare_versions(
        self, version_a_id: str, version_b_id: str
    ) -> VersionDiff:
        """Compare two version snapshots and produce a detailed diff.

        *version_a* is treated as the baseline ("before") and *version_b*
        as the current state ("after").
        """
        dir_a = osp.join(
            self._versions_root, version_a_id, "annotations"
        )
        dir_b = osp.join(
            self._versions_root, version_b_id, "annotations"
        )

        files_a = self._list_json_files(dir_a)
        files_b = self._list_json_files(dir_b)

        all_names = sorted(set(files_a.keys()) | set(files_b.keys()))

        image_diffs: List[ImageDiff] = []
        total_added_images = 0
        total_removed_images = 0
        total_modified_images = 0
        total_unchanged_images = 0
        total_added_shapes = 0
        total_removed_shapes = 0
        total_modified_shapes = 0

        for name in all_names:
            in_a = name in files_a
            in_b = name in files_b

            if in_a and not in_b:
                # Image removed in version B
                data_a = _load_annotation(files_a[name])
                shapes_a = data_a.get("shapes", []) if data_a else []
                removed_count = len(shapes_a)
                details = [
                    {
                        "label": s.get("label", ""),
                        "shape_type": s.get("shape_type", ""),
                        "change_type": "removed",
                    }
                    for s in shapes_a
                ]
                diff = ImageDiff(
                    image_name=name,
                    status="removed",
                    removed_shapes=removed_count,
                    details=details,
                )
                total_removed_images += 1
                total_removed_shapes += removed_count

            elif in_b and not in_a:
                # Image added in version B
                data_b = _load_annotation(files_b[name])
                shapes_b = data_b.get("shapes", []) if data_b else []
                added_count = len(shapes_b)
                details = [
                    {
                        "label": s.get("label", ""),
                        "shape_type": s.get("shape_type", ""),
                        "change_type": "added",
                    }
                    for s in shapes_b
                ]
                diff = ImageDiff(
                    image_name=name,
                    status="added",
                    added_shapes=added_count,
                    details=details,
                )
                total_added_images += 1
                total_added_shapes += added_count

            else:
                # Image present in both versions -- compare shapes
                data_a = _load_annotation(files_a[name])
                data_b = _load_annotation(files_b[name])
                shapes_a = data_a.get("shapes", []) if data_a else []
                shapes_b = data_b.get("shapes", []) if data_b else []

                added, removed, modified, details = _compare_shape_lists(
                    shapes_a, shapes_b
                )

                if added == 0 and removed == 0 and modified == 0:
                    status = "unchanged"
                    total_unchanged_images += 1
                else:
                    status = "modified"
                    total_modified_images += 1

                total_added_shapes += added
                total_removed_shapes += removed
                total_modified_shapes += modified

                diff = ImageDiff(
                    image_name=name,
                    status=status,
                    added_shapes=added,
                    removed_shapes=removed,
                    modified_shapes=modified,
                    details=details,
                )

            image_diffs.append(diff)

        summary = {
            "added_images": total_added_images,
            "removed_images": total_removed_images,
            "modified_images": total_modified_images,
            "unchanged_images": total_unchanged_images,
            "added_shapes": total_added_shapes,
            "removed_shapes": total_removed_shapes,
            "modified_shapes": total_modified_shapes,
        }

        return VersionDiff(
            version_a=version_a_id,
            version_b=version_b_id,
            image_diffs=image_diffs,
            summary=summary,
        )

    # -- private helpers ----------------------------------------------------

    @staticmethod
    def _read_metadata(meta_path: str) -> Optional[VersionInfo]:
        """Parse a ``metadata.json`` file into a `VersionInfo`."""
        try:
            with open(meta_path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            return VersionInfo(
                version_id=data["version_id"],
                name=data["name"],
                description=data["description"],
                timestamp=data["timestamp"],
                stats=data["stats"],
                size_bytes=data["size_bytes"],
            )
        except (OSError, json.JSONDecodeError, KeyError) as exc:
            logger.warning(
                "Failed to read metadata from '%s': %s", meta_path, exc
            )
            return None

    @staticmethod
    def _compute_annotation_stats(
        annotations_dir: str, image_list: List[str]
    ) -> Dict:
        """Compute summary statistics for a set of annotations.

        Parameters
        ----------
        annotations_dir:
            Directory containing the copied ``.json`` label files.
        image_list:
            Full list of image filenames in the project (used for the
            ``image_count`` stat).

        Returns
        -------
        dict
            Keys: ``image_count``, ``annotated_image_count``,
            ``total_shapes``, ``class_distribution``.
        """
        image_count = len(image_list)
        annotated_image_count = 0
        total_shapes = 0
        class_distribution: Dict[str, int] = {}

        try:
            json_files = [
                f
                for f in os.listdir(annotations_dir)
                if f.lower().endswith(".json")
            ]
        except OSError as exc:
            logger.warning(
                "Cannot list annotations directory '%s': %s",
                annotations_dir,
                exc,
            )
            return {
                "image_count": image_count,
                "annotated_image_count": 0,
                "total_shapes": 0,
                "class_distribution": {},
            }

        for json_file in json_files:
            json_path = osp.join(annotations_dir, json_file)
            data = _load_annotation(json_path)
            if data is None:
                continue

            shapes = data.get("shapes", [])
            if shapes:
                annotated_image_count += 1
                total_shapes += len(shapes)
                for shape in shapes:
                    label = shape.get("label", "")
                    class_distribution[label] = (
                        class_distribution.get(label, 0) + 1
                    )

        return {
            "image_count": image_count,
            "annotated_image_count": annotated_image_count,
            "total_shapes": total_shapes,
            "class_distribution": class_distribution,
        }

    @staticmethod
    def _compute_dir_size(dir_path: str) -> int:
        """Return total size in bytes of all files under *dir_path*."""
        total = 0
        try:
            for dirpath, _dirnames, filenames in os.walk(dir_path):
                for fname in filenames:
                    fpath = osp.join(dirpath, fname)
                    try:
                        total += osp.getsize(fpath)
                    except OSError:
                        pass
        except OSError as exc:
            logger.warning(
                "Failed to compute size of '%s': %s", dir_path, exc
            )
        return total

    @staticmethod
    def _list_json_files(directory: str) -> Dict[str, str]:
        """Return a mapping of ``{basename: full_path}`` for ``.json`` files."""
        result: Dict[str, str] = {}
        if not osp.isdir(directory):
            return result
        try:
            for fname in os.listdir(directory):
                if fname.lower().endswith(".json"):
                    result[fname] = osp.join(directory, fname)
        except OSError as exc:
            logger.warning(
                "Failed to list JSON files in '%s': %s", directory, exc
            )
        return result
