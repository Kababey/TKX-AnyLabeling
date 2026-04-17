"""Train/test/val split management module for X-AnyLabeling.

Provides functionality to partition images into train/test/val splits,
persist assignments to disk, synchronize with the current image list,
and compute per-split statistics including class distribution.
"""

import json
import math
import os
import random
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

from anylabeling.views.labeling.logger import logger


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def _read_xlabel_labels(json_path: str) -> List[str]:
    """Read an XLABEL JSON annotation file and return label names.

    Args:
        json_path: Absolute path to the annotation JSON file.

    Returns:
        List of label name strings extracted from ``shapes``.
        Returns an empty list when the file is missing, corrupt,
        or does not contain a ``shapes`` key.
    """
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        shapes = data.get("shapes", [])
        if not isinstance(shapes, list):
            return []
        return [
            s["label"]
            for s in shapes
            if isinstance(s, dict) and "label" in s
        ]
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as e:
        logger.debug(
            "Could not read labels from %s: %s", json_path, e
        )
        return []


def _get_label_file_path(
    image_filename: str, label_dir: str
) -> Optional[str]:
    """Find the corresponding .json label file for an image.

    The label file is assumed to share the same stem as the image file
    and reside in *label_dir*.

    Args:
        image_filename: Image file name (not a full path).
        label_dir: Directory containing annotation JSON files.

    Returns:
        Absolute path to the label file if it exists, otherwise ``None``.
    """
    stem = os.path.splitext(image_filename)[0]
    label_path = os.path.join(label_dir, stem + ".json")
    if os.path.isfile(label_path):
        return label_path
    return None


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class SplitStats:
    """Statistics for a single split partition."""

    split_name: str
    image_count: int
    shape_count: int
    class_distribution: Dict[str, int] = field(default_factory=dict)
    percentage: float = 0.0


# ---------------------------------------------------------------------------
# SplitManager
# ---------------------------------------------------------------------------


class SplitManager:
    """Manage train / val / test / unassigned image splits.

    Splits are persisted as a JSON file (``.xanylabeling_splits.json``)
    inside the project directory (or an optional *output_dir*).
    """

    DEFAULT_RATIOS: Dict[str, float] = {
        "train": 0.7,
        "val": 0.2,
        "test": 0.1,
    }
    SPLIT_NAMES: List[str] = ["train", "val", "test", "unassigned"]

    _SPLITS_FILENAME = ".xanylabeling_splits.json"
    _FILE_VERSION = "1.0"

    # ---- construction / persistence ----

    def __init__(
        self, project_dir: str, output_dir: Optional[str] = None
    ) -> None:
        """Initialise a SplitManager.

        Args:
            project_dir: Root directory of the labelling project.
            output_dir: Optional directory for the splits file.
                        Falls back to *project_dir* when ``None``.
        """
        self._project_dir = project_dir
        base_dir = output_dir if output_dir else project_dir
        self._splits_path = os.path.join(base_dir, self._SPLITS_FILENAME)
        self._splits: Dict[str, List[str]] = {
            name: [] for name in self.SPLIT_NAMES
        }
        self._ratios: Dict[str, float] = dict(self.DEFAULT_RATIOS)
        self._created: Optional[str] = None
        self._last_modified: Optional[str] = None

        # Attempt to restore a previous session.
        self.load_splits()

    # ---- persistence ----

    def load_splits(self) -> bool:
        """Load split assignments from the JSON file on disk.

        Returns:
            ``True`` if the file was found and parsed successfully,
            ``False`` otherwise (internal state is reset to empty).
        """
        if not os.path.isfile(self._splits_path):
            return False

        try:
            with open(self._splits_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            splits_data = data.get("splits", {})
            loaded: Dict[str, List[str]] = {
                name: [] for name in self.SPLIT_NAMES
            }
            for name in self.SPLIT_NAMES:
                entries = splits_data.get(name, [])
                if isinstance(entries, list):
                    loaded[name] = [
                        str(e) for e in entries if isinstance(e, str)
                    ]

            self._splits = loaded

            metadata = data.get("metadata", {})
            self._created = metadata.get("created")
            self._last_modified = metadata.get("last_modified")
            ratios = metadata.get("ratios")
            if isinstance(ratios, dict):
                self._ratios = {
                    k: float(v) for k, v in ratios.items()
                    if isinstance(v, (int, float))
                }

            logger.info(
                "Loaded splits from %s (%d images total)",
                self._splits_path,
                sum(len(v) for v in self._splits.values()),
            )
            return True

        except (OSError, json.JSONDecodeError, ValueError, TypeError) as e:
            logger.warning(
                "Failed to load splits file %s: %s", self._splits_path, e
            )
            self._splits = {name: [] for name in self.SPLIT_NAMES}
            return False

    def save_splits(self) -> None:
        """Persist current split assignments to disk."""
        now = datetime.now().isoformat(timespec="seconds")
        if self._created is None:
            self._created = now
        self._last_modified = now

        payload = {
            "version": self._FILE_VERSION,
            "splits": {
                name: list(self._splits.get(name, []))
                for name in self.SPLIT_NAMES
            },
            "metadata": {
                "created": self._created,
                "last_modified": self._last_modified,
                "ratios": dict(self._ratios),
            },
        }

        try:
            directory = os.path.dirname(self._splits_path)
            if directory and not os.path.isdir(directory):
                os.makedirs(directory, exist_ok=True)

            with open(self._splits_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, ensure_ascii=False)

            logger.info("Saved splits to %s", self._splits_path)
        except OSError as e:
            logger.error(
                "Failed to save splits file %s: %s", self._splits_path, e
            )

    # ---- query ----

    def get_partition(self, filename: str) -> str:
        """Return the partition name for *filename*.

        Args:
            filename: Image filename (basename only).

        Returns:
            One of ``"train"``, ``"val"``, ``"test"``, or
            ``"unassigned"``.
        """
        for name in self.SPLIT_NAMES:
            if filename in self._splits[name]:
                return name
        return "unassigned"

    def get_split(self, split_name: str) -> List[str]:
        """Return the list of filenames assigned to *split_name*.

        Args:
            split_name: Name of the split (e.g. ``"train"``).

        Returns:
            Copy of the filename list for the requested split.
            Returns an empty list if the name is unknown.
        """
        return list(self._splits.get(split_name, []))

    def get_all_splits(self) -> Dict[str, List[str]]:
        """Return a copy of all split assignments."""
        return {k: list(v) for k, v in self._splits.items()}

    def has_splits(self) -> bool:
        """Whether any non-unassigned split contains images."""
        return any(
            len(self._splits.get(name, [])) > 0
            for name in self.SPLIT_NAMES
            if name != "unassigned"
        )

    def get_splits_file_path(self) -> str:
        """Return the absolute path of the splits JSON file."""
        return self._splits_path

    # ---- mutation ----

    def set_partition(self, filename: str, partition: str) -> None:
        """Assign *filename* to *partition*, removing it from any other.

        Args:
            filename: Image filename (basename only).
            partition: Target partition name.

        Raises:
            ValueError: If *partition* is not a recognised split name.
        """
        if partition not in self.SPLIT_NAMES:
            raise ValueError(
                f"Unknown partition '{partition}'. "
                f"Must be one of {self.SPLIT_NAMES}"
            )
        # Remove from wherever it currently lives.
        for name in self.SPLIT_NAMES:
            try:
                self._splits[name].remove(filename)
            except ValueError:
                pass
        self._splits[partition].append(filename)

    def set_partitions_batch(
        self, filenames: List[str], partition: str
    ) -> None:
        """Batch-assign multiple filenames to *partition*.

        Args:
            filenames: List of image filenames.
            partition: Target partition name.

        Raises:
            ValueError: If *partition* is not a recognised split name.
        """
        if partition not in self.SPLIT_NAMES:
            raise ValueError(
                f"Unknown partition '{partition}'. "
                f"Must be one of {self.SPLIT_NAMES}"
            )
        # Build a set for fast removal.
        to_move = set(filenames)

        for name in self.SPLIT_NAMES:
            self._splits[name] = [
                f for f in self._splits[name] if f not in to_move
            ]

        self._splits[partition].extend(filenames)

    def sync_with_image_list(self, image_filenames: List[str]) -> None:
        """Synchronise splits with the current project image list.

        * New images (present in *image_filenames* but absent from all
          splits) are added to ``"unassigned"``.
        * Images no longer in *image_filenames* are removed from their
          split.

        Args:
            image_filenames: Authoritative list of image basenames.
        """
        current_set = set(image_filenames)

        # Collect every filename already tracked.
        known: Dict[str, str] = {}
        for name in self.SPLIT_NAMES:
            for fn in self._splits[name]:
                known[fn] = name

        # Remove stale entries.
        for name in self.SPLIT_NAMES:
            self._splits[name] = [
                f for f in self._splits[name] if f in current_set
            ]

        # Add new images.
        already_tracked = set(known.keys()) & current_set
        for fn in image_filenames:
            if fn not in already_tracked:
                self._splits["unassigned"].append(fn)

    def clear_splits(self) -> None:
        """Move every image to ``"unassigned"``."""
        all_files: List[str] = []
        for name in self.SPLIT_NAMES:
            all_files.extend(self._splits[name])
            self._splits[name] = []
        self._splits["unassigned"] = all_files

    # ---- auto-split ----

    def auto_split(
        self,
        image_list: List[str],
        ratios: Optional[Dict[str, float]] = None,
        strategy: str = "random",
        label_dir: Optional[str] = None,
        seed: Optional[int] = None,
    ) -> None:
        """Automatically split *image_list* into train/val/test.

        Args:
            image_list: List of image filenames to split.
            ratios: Mapping of split name to ratio.  Must sum to ~1.0.
                    Defaults to :pyattr:`DEFAULT_RATIOS`.
            strategy: ``"random"`` or ``"stratified"``.
            label_dir: Directory of annotation JSONs (required for
                       ``"stratified"``).
            seed: Optional RNG seed for reproducibility.

        Raises:
            ValueError: If *ratios* do not approximately sum to 1.0 or
                        contain unknown split names, or if *strategy* is
                        unrecognised.
        """
        if ratios is None:
            ratios = dict(self.DEFAULT_RATIOS)

        # --- validate ratios ---
        for key in ratios:
            if key not in self.SPLIT_NAMES or key == "unassigned":
                raise ValueError(
                    f"Ratio key '{key}' is not a valid split name. "
                    f"Use one of {[n for n in self.SPLIT_NAMES if n != 'unassigned']}"
                )

        total = sum(ratios.values())
        if not math.isclose(total, 1.0, abs_tol=1e-3):
            raise ValueError(
                f"Ratios must sum to ~1.0, got {total:.4f}"
            )

        # --- validate strategy ---
        if strategy not in ("random", "stratified"):
            raise ValueError(
                f"Unknown strategy '{strategy}'. "
                "Use 'random' or 'stratified'."
            )

        # Store ratios for metadata.
        self._ratios = dict(ratios)

        if strategy == "stratified":
            if label_dir is None:
                logger.warning(
                    "label_dir not provided for stratified split, "
                    "falling back to random."
                )
                self._random_split(image_list, ratios, seed)
            else:
                self._stratified_split(image_list, label_dir, ratios, seed)
        else:
            self._random_split(image_list, ratios, seed)

    # ---- private split implementations ----

    def _random_split(
        self,
        image_list: List[str],
        ratios: Dict[str, float],
        seed: Optional[int],
    ) -> None:
        """Simple random split by ratios."""
        rng = random.Random(seed)
        shuffled = list(image_list)
        rng.shuffle(shuffled)

        assignments = self._partition_by_ratios(shuffled, ratios)
        self._apply_assignments(assignments, image_list)

    def _stratified_split(
        self,
        image_list: List[str],
        label_dir: str,
        ratios: Dict[str, float],
        seed: Optional[int],
    ) -> None:
        """Stratified split ensuring similar class distribution.

        Images are grouped by their *primary label* (the most frequent
        label in their annotation file).  Within each group the items
        are split proportionally according to *ratios*.

        If annotations cannot be read for any image it is placed into a
        fallback group and split randomly.
        """
        rng = random.Random(seed)

        # Determine primary label per image.
        groups: Dict[str, List[str]] = defaultdict(list)
        for img in image_list:
            label_path = _get_label_file_path(img, label_dir)
            if label_path is None:
                groups["__unlabeled__"].append(img)
                continue
            labels = _read_xlabel_labels(label_path)
            if not labels:
                groups["__unlabeled__"].append(img)
                continue

            # Primary label = most frequent label in the file.
            freq: Dict[str, int] = defaultdict(int)
            for lbl in labels:
                freq[lbl] += 1
            primary = max(freq, key=lambda k: freq[k])
            groups[primary].append(img)

        # Within each group, split proportionally.
        all_assignments: Dict[str, List[str]] = {
            name: [] for name in self.SPLIT_NAMES
        }

        for _group_key, members in groups.items():
            rng.shuffle(members)
            group_assignments = self._partition_by_ratios(members, ratios)
            for name, items in group_assignments.items():
                all_assignments[name].extend(items)

        self._apply_assignments(all_assignments, image_list)

    @staticmethod
    def _partition_by_ratios(
        items: List[str], ratios: Dict[str, float]
    ) -> Dict[str, List[str]]:
        """Divide *items* into buckets according to *ratios*.

        The last bucket receives any remainder so that all items are
        assigned exactly once.
        """
        n = len(items)
        result: Dict[str, List[str]] = {}
        start = 0
        ordered_names = list(ratios.keys())

        for i, name in enumerate(ordered_names):
            if i == len(ordered_names) - 1:
                # Last bucket gets the remainder.
                result[name] = items[start:]
            else:
                count = round(n * ratios[name])
                result[name] = items[start : start + count]
                start += count

        return result

    def _apply_assignments(
        self,
        assignments: Dict[str, List[str]],
        full_image_list: List[str],
    ) -> None:
        """Write *assignments* into ``_splits``, putting leftovers
        into ``"unassigned"``.
        """
        assigned_set: set = set()
        for name in self.SPLIT_NAMES:
            self._splits[name] = []

        for name, items in assignments.items():
            if name in self.SPLIT_NAMES:
                self._splits[name] = list(items)
                assigned_set.update(items)

        # Anything not assigned goes to unassigned.
        unassigned = [f for f in full_image_list if f not in assigned_set]
        self._splits["unassigned"].extend(unassigned)

    # ---- statistics ----

    def get_split_stats(self, label_dir: str) -> List[SplitStats]:
        """Compute statistics for every split.

        Args:
            label_dir: Directory containing annotation JSON files.

        Returns:
            List of :class:`SplitStats`, one per split name.
        """
        total_images = sum(len(v) for v in self._splits.values())
        stats: List[SplitStats] = []

        for name in self.SPLIT_NAMES:
            filenames = self._splits.get(name, [])
            image_count = len(filenames)
            shape_count = 0
            class_dist: Dict[str, int] = defaultdict(int)

            for img in filenames:
                label_path = _get_label_file_path(img, label_dir)
                if label_path is None:
                    continue
                labels = _read_xlabel_labels(label_path)
                shape_count += len(labels)
                for lbl in labels:
                    class_dist[lbl] += 1

            percentage = (
                (image_count / total_images * 100.0)
                if total_images > 0
                else 0.0
            )

            stats.append(
                SplitStats(
                    split_name=name,
                    image_count=image_count,
                    shape_count=shape_count,
                    class_distribution=dict(class_dist),
                    percentage=round(percentage, 2),
                )
            )

        return stats

    def get_overall_stats(self, label_dir: str) -> Dict:
        """Return aggregate statistics across all splits.

        Args:
            label_dir: Directory containing annotation JSON files.

        Returns:
            Dictionary with ``total_images``, ``total_shapes``,
            ``class_distribution``, and per-split ``splits`` summaries.
        """
        split_stats = self.get_split_stats(label_dir)

        total_images = sum(s.image_count for s in split_stats)
        total_shapes = sum(s.shape_count for s in split_stats)
        overall_dist: Dict[str, int] = defaultdict(int)

        splits_summary: Dict[str, Dict] = {}
        for s in split_stats:
            for lbl, cnt in s.class_distribution.items():
                overall_dist[lbl] += cnt
            splits_summary[s.split_name] = {
                "image_count": s.image_count,
                "shape_count": s.shape_count,
                "percentage": s.percentage,
            }

        return {
            "total_images": total_images,
            "total_shapes": total_shapes,
            "class_distribution": dict(overall_dist),
            "splits": splits_summary,
        }
