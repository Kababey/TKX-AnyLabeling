"""Multi-project management module.

Supports multiple dataset projects, each defined by a
``.xanylabeling_project.json`` file at its root. Maintains a registry
of recently opened projects at ``~/.xanylabeling/projects_registry.json``.

Safety: this module NEVER deletes user directories. Project "close"
and "remove from recent" operations only affect the registry.
"""

import copy
import json
import os
import os.path as osp
import tempfile
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

from anylabeling.views.labeling.logger import logger


PROJECT_FILE_NAME = ".xanylabeling_project.json"
REGISTRY_DIR_NAME = ".xanylabeling"
REGISTRY_FILE_NAME = "projects_registry.json"
SCHEMA_VERSION = "1.0"
MAX_RECENT_PROJECTS = 50

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff",
}

# Default 20-color palette used when auto-assigning class colors
_DEFAULT_PALETTE = [
    "#e6194B", "#3cb44b", "#ffe119", "#4363d8", "#f58231",
    "#42d4f4", "#f032e6", "#fabed4", "#469990", "#dcbeff",
    "#9A6324", "#800000", "#aaffc3", "#808000", "#ffd8b1",
    "#000075", "#a9a9a9", "#911eb4", "#bfef45", "#808080",
]


@dataclass
class ProjectInfo:
    """In-memory representation of a project config."""

    name: str
    description: str
    path: str  # absolute path to the project root directory
    created: str
    modified: str
    version: str = "1.0"
    stats: Dict = field(default_factory=dict)
    settings: Dict = field(default_factory=dict)
    classes: List[Dict] = field(default_factory=list)
    paths: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        """Serialize to the JSON schema (excludes ``path``)."""
        return {
            "schema_version": SCHEMA_VERSION,
            "name": self.name,
            "description": self.description,
            "created": self.created,
            "modified": self.modified,
            "version": self.version,
            "paths": self.paths or {
                "images_dir": "images",
                "annotations_dir": "annotations",
            },
            "settings": self.settings,
            "classes": self.classes,
            "stats": self.stats,
        }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _atomic_write_json(path: str, data: Dict) -> None:
    """Write JSON atomically: write to temp file in same dir, then rename."""
    parent = osp.dirname(path) or "."
    os.makedirs(parent, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(
        prefix=".tmp_",
        suffix=".json",
        dir=parent,
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
            f.flush()
            try:
                os.fsync(f.fileno())
            except OSError:
                pass
        os.replace(tmp_path, path)
    except Exception:
        # Clean up temp file on failure
        try:
            if osp.exists(tmp_path):
                os.remove(tmp_path)
        except OSError:
            pass
        raise


def _read_json(path: str) -> Optional[Dict]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        logger.debug("Failed to read JSON '%s': %s", path, exc)
        return None


def _normalize_classes(classes) -> List[Dict]:
    """Accept list of strings or dicts and normalize to dicts with colors."""
    if not classes:
        return []
    result: List[Dict] = []
    seen = set()
    for i, entry in enumerate(classes):
        if isinstance(entry, str):
            name = entry
            color = _DEFAULT_PALETTE[i % len(_DEFAULT_PALETTE)]
        elif isinstance(entry, dict) and "name" in entry:
            name = entry["name"]
            color = entry.get(
                "color", _DEFAULT_PALETTE[i % len(_DEFAULT_PALETTE)]
            )
        else:
            continue
        if name in seen:
            continue
        seen.add(name)
        result.append({"name": name, "color": color})
    return result


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _looks_like_dataset(path: str) -> bool:
    """Heuristic: directory appears to contain image data."""
    if not osp.isdir(path):
        return False
    try:
        entries = os.listdir(path)
    except OSError:
        return False
    # Check for images subdir or loose images at top level
    if "images" in entries and osp.isdir(osp.join(path, "images")):
        return True
    for e in entries[:50]:
        ext = osp.splitext(e)[1].lower()
        if ext in IMAGE_EXTENSIONS:
            return True
    return False


# ---------------------------------------------------------------------------
# ProjectManager
# ---------------------------------------------------------------------------


class ProjectManager:
    """Manage multi-project state: project configs plus recent-projects registry."""

    def __init__(self, workspace_dir: Optional[str] = None) -> None:
        if workspace_dir is None:
            workspace_dir = osp.join(
                osp.expanduser("~"), REGISTRY_DIR_NAME
            )
        self._workspace_dir = workspace_dir
        self._registry_path = osp.join(workspace_dir, REGISTRY_FILE_NAME)

    @property
    def workspace_dir(self) -> str:
        return self._workspace_dir

    @property
    def registry_path(self) -> str:
        return self._registry_path

    # ---- project file I/O ----

    def load_project(self, project_dir: str) -> ProjectInfo:
        """Load a project from its directory. Raises ValueError if invalid."""
        cfg_path = osp.join(project_dir, PROJECT_FILE_NAME)
        data = _read_json(cfg_path)
        if data is None:
            raise ValueError(
                f"Not a valid X-AnyLabeling project: {project_dir}"
            )
        try:
            info = ProjectInfo(
                name=str(data.get("name", osp.basename(project_dir))),
                description=str(data.get("description", "")),
                path=osp.abspath(project_dir),
                created=str(data.get("created", _now_iso())),
                modified=str(data.get("modified", _now_iso())),
                version=str(data.get("version", "1.0")),
                stats=dict(data.get("stats", {})),
                settings=dict(data.get("settings", {})),
                classes=_normalize_classes(data.get("classes", [])),
                paths=dict(data.get("paths", {})),
            )
            return info
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Corrupt project config at {cfg_path}: {exc}"
            ) from exc

    def save_project(self, project_info: ProjectInfo) -> None:
        """Persist project config atomically."""
        project_info.modified = _now_iso()
        cfg_path = osp.join(project_info.path, PROJECT_FILE_NAME)
        _atomic_write_json(cfg_path, project_info.to_dict())
        logger.info("Saved project config: %s", cfg_path)

    def create_project(
        self,
        project_dir: str,
        name: str,
        description: str = "",
        classes: Optional[List] = None,
        settings: Optional[Dict] = None,
    ) -> ProjectInfo:
        """Create a new project in the given directory.

        Does NOT create ``project_dir`` itself — caller must ensure it
        exists. Raises ValueError if the directory is already a project.
        """
        if not osp.isdir(project_dir):
            raise ValueError(
                f"Project directory does not exist: {project_dir}"
            )
        cfg_path = osp.join(project_dir, PROJECT_FILE_NAME)
        if osp.exists(cfg_path):
            raise ValueError(
                f"Directory is already a project: {project_dir}"
            )

        now = _now_iso()
        default_settings = {
            "target_resolution": [0, 0],  # 0 means "not set yet"
            "resize_mode": "letterbox",
            "annotation_format": "xlabel",
            "auto_resize_new_images": False,
        }
        if settings:
            default_settings.update(settings)

        info = ProjectInfo(
            name=name or osp.basename(project_dir),
            description=description,
            path=osp.abspath(project_dir),
            created=now,
            modified=now,
            version="1.0",
            stats={
                "image_count": 0,
                "annotated_count": 0,
                "total_shapes": 0,
                "last_updated": now,
            },
            settings=default_settings,
            classes=_normalize_classes(classes),
            paths={
                "images_dir": "images",
                "annotations_dir": "annotations",
            },
        )
        self.save_project(info)
        self.add_to_recent(info.path, info)
        return info

    def open_project(self, project_dir: str) -> ProjectInfo:
        """Load project and register as recently opened."""
        info = self.load_project(project_dir)
        self.add_to_recent(info.path, info)
        return info

    def close_project(self, project_info: ProjectInfo) -> None:
        """In-memory close. Does NOT touch any files on disk."""
        logger.debug("Closed project (in-memory): %s", project_info.path)

    def is_project_dir(self, path: str) -> bool:
        return osp.isfile(osp.join(path, PROJECT_FILE_NAME))

    def detect_project_from_directory(
        self, path: str
    ) -> Optional[ProjectInfo]:
        """Return ProjectInfo if directory is a project, else None."""
        if not self.is_project_dir(path):
            return None
        try:
            return self.load_project(path)
        except ValueError:
            return None

    # ---- stats / classes / settings updates ----

    def update_stats(
        self,
        project_info: ProjectInfo,
        image_count: int,
        annotated_count: int,
        total_shapes: int,
    ) -> None:
        project_info.stats = {
            "image_count": int(image_count),
            "annotated_count": int(annotated_count),
            "total_shapes": int(total_shapes),
            "last_updated": _now_iso(),
        }
        self.save_project(project_info)

    def update_classes(
        self, project_info: ProjectInfo, classes: List
    ) -> None:
        project_info.classes = _normalize_classes(classes)
        self.save_project(project_info)

    def update_settings(
        self, project_info: ProjectInfo, settings: Dict
    ) -> None:
        merged = dict(project_info.settings or {})
        merged.update(settings or {})
        project_info.settings = merged
        self.save_project(project_info)

    # ---- path helpers ----

    def get_images_dir(self, project_info: ProjectInfo) -> str:
        sub = project_info.paths.get("images_dir", "images")
        return osp.join(project_info.path, sub)

    def get_annotations_dir(self, project_info: ProjectInfo) -> str:
        sub = project_info.paths.get("annotations_dir", "annotations")
        return osp.join(project_info.path, sub)

    # ---- registry ----

    def _load_registry(self) -> Dict:
        data = _read_json(self._registry_path)
        if not isinstance(data, dict):
            return {"recent_projects": []}
        if "recent_projects" not in data or not isinstance(
            data["recent_projects"], list
        ):
            data["recent_projects"] = []
        return data

    def _save_registry(self, data: Dict) -> None:
        try:
            _atomic_write_json(self._registry_path, data)
        except OSError as exc:
            logger.warning("Failed to save registry: %s", exc)

    def list_recent_projects(self) -> List[Dict]:
        """Return list of recent project entries, each with an 'exists' flag."""
        data = self._load_registry()
        entries = data.get("recent_projects", [])
        out = []
        for e in entries:
            if not isinstance(e, dict) or "path" not in e:
                continue
            path = e["path"]
            entry = dict(e)
            entry["exists"] = osp.isfile(osp.join(path, PROJECT_FILE_NAME))
            out.append(entry)
        return out

    def add_to_recent(
        self, project_dir: str, project_info: ProjectInfo
    ) -> None:
        abs_path = osp.abspath(project_dir)
        now = _now_iso()
        data = self._load_registry()
        entries = data.get("recent_projects", [])
        # Remove existing entry for same path (case-insensitive on Windows)
        norm = osp.normcase(abs_path)
        entries = [
            e for e in entries
            if not (
                isinstance(e, dict)
                and osp.normcase(e.get("path", "")) == norm
            )
        ]
        entries.insert(
            0,
            {
                "path": abs_path,
                "name": project_info.name,
                "last_opened": now,
                "description": project_info.description,
            },
        )
        # Cap list
        entries = entries[:MAX_RECENT_PROJECTS]
        data["recent_projects"] = entries
        self._save_registry(data)

    def remove_from_recent(self, project_dir: str) -> None:
        abs_path = osp.abspath(project_dir)
        norm = osp.normcase(abs_path)
        data = self._load_registry()
        data["recent_projects"] = [
            e for e in data.get("recent_projects", [])
            if not (
                isinstance(e, dict)
                and osp.normcase(e.get("path", "")) == norm
            )
        ]
        self._save_registry(data)

    def clear_recent(self) -> None:
        self._save_registry({"recent_projects": []})

    # ---- discovery ----

    def scan_for_projects(
        self, search_dir: str, max_depth: int = 3
    ) -> List[str]:
        """Find project directories under ``search_dir`` up to ``max_depth``."""
        if not osp.isdir(search_dir):
            return []
        found: List[str] = []
        stack = [(search_dir, 0)]
        skip_dirs = {
            ".git", ".hg", ".svn", "__pycache__", "node_modules",
            ".venv", "venv", ".idea", ".vscode", ".xanylabeling",
        }
        while stack:
            path, depth = stack.pop()
            if self.is_project_dir(path):
                found.append(osp.abspath(path))
                continue  # don't descend into projects
            if depth >= max_depth:
                continue
            try:
                for entry in os.listdir(path):
                    if entry.startswith(".") or entry in skip_dirs:
                        continue
                    full = osp.join(path, entry)
                    if osp.isdir(full):
                        stack.append((full, depth + 1))
            except (OSError, PermissionError):
                continue
        return sorted(found)
