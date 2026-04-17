"""Class management dialog - rename, merge, delete classes across
all annotations in the dataset.
"""

import json
import os
import os.path as osp
import tempfile
from typing import Dict, List, Optional

from PyQt6 import QtCore
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QColorDialog,
    QDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QGroupBox,
    QHeaderView,
    QInputDialog,
    QMenu,
)

from anylabeling.views.labeling.logger import logger
from anylabeling.views.labeling.utils.style import (
    get_cancel_btn_style,
    get_dialog_style,
    get_ok_btn_style,
)


def _atomic_write_json(path: str, data: Dict) -> None:
    parent = osp.dirname(path) or "."
    fd, tmp = tempfile.mkstemp(prefix=".tmp_", suffix=".json", dir=parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
            f.flush()
            try:
                os.fsync(f.fileno())
            except OSError:
                pass
        os.replace(tmp, path)
    except Exception:
        try:
            if osp.exists(tmp):
                os.remove(tmp)
        except OSError:
            pass
        raise


class ClassRenameThread(QThread):
    """Apply a class rename/merge/delete across all annotation JSONs."""

    # action: "rename" | "merge" | "delete"
    # For rename: from_label -> to_label (single mapping)
    # For merge:  many from_labels -> single to_label
    # For delete: drop all shapes with label in from_labels

    progress = pyqtSignal(int, int)
    finished = pyqtSignal(bool, str, int, int)  # success, error, files_changed, shapes_changed

    def __init__(
        self,
        annotation_paths: List[str],
        action: str,
        from_labels: List[str],
        to_label: Optional[str] = None,
    ):
        super().__init__()
        self.annotation_paths = annotation_paths
        self.action = action
        self.from_labels = set(from_labels)
        self.to_label = to_label
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    def run(self):
        files_changed = 0
        shapes_changed = 0
        total = len(self.annotation_paths)
        try:
            for i, path in enumerate(self.annotation_paths, 1):
                if self._cancelled:
                    break
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                except (OSError, json.JSONDecodeError) as exc:
                    logger.warning("Skipping '%s': %s", path, exc)
                    self.progress.emit(i, total)
                    continue

                shapes = data.get("shapes", [])
                if not isinstance(shapes, list):
                    self.progress.emit(i, total)
                    continue

                file_shape_changes = 0
                if self.action == "delete":
                    before = len(shapes)
                    shapes = [
                        s for s in shapes
                        if s.get("label") not in self.from_labels
                    ]
                    file_shape_changes = before - len(shapes)
                    data["shapes"] = shapes
                elif self.action in ("rename", "merge"):
                    for s in shapes:
                        if s.get("label") in self.from_labels:
                            s["label"] = self.to_label
                            file_shape_changes += 1

                if file_shape_changes > 0:
                    try:
                        _atomic_write_json(path, data)
                        files_changed += 1
                        shapes_changed += file_shape_changes
                    except OSError as exc:
                        logger.warning(
                            "Failed to write '%s': %s", path, exc
                        )

                self.progress.emit(i, total)

            self.finished.emit(True, "", files_changed, shapes_changed)
        except Exception as exc:
            logger.error("ClassRenameThread failed: %s", exc)
            self.finished.emit(False, str(exc), files_changed, shapes_changed)


class ClassManagerDialog(QDialog):
    """Manage classes: rename, merge, delete, set colors."""

    classes_changed = pyqtSignal(list)  # emits updated class list

    def __init__(
        self,
        classes: List[Dict],
        annotation_paths: List[str],
        parent=None,
    ):
        super().__init__(parent)
        self._classes = [dict(c) for c in classes]
        self._annotation_paths = annotation_paths
        self._thread: Optional[ClassRenameThread] = None
        self._dirty = False

        self.setWindowTitle(self.tr("Class Manager"))
        self.setMinimumSize(640, 480)
        self.setStyleSheet(get_dialog_style())

        self._build_ui()
        self._refresh()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(12)

        header = QLabel(self.tr("Classes in this project"))
        header.setStyleSheet("font-size: 16px; font-weight: 600;")
        layout.addWidget(header)

        hint = QLabel(self.tr(
            "Right-click a class for actions (rename, merge, delete). "
            "Class color can be changed by double-clicking the color cell."
        ))
        hint.setStyleSheet("color: #888;")
        hint.setWordWrap(True)
        layout.addWidget(hint)

        # Table
        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels([
            self.tr("Class name"),
            self.tr("Color"),
            self.tr("Instances"),
        ])
        self.table.setEditTriggers(
            QAbstractItemView.EditTrigger.NoEditTriggers
        )
        self.table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows
        )
        self.table.setSelectionMode(
            QAbstractItemView.SelectionMode.ExtendedSelection
        )
        self.table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeMode.Stretch
        )
        self.table.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.ResizeMode.ResizeToContents
        )
        self.table.horizontalHeader().setSectionResizeMode(
            2, QHeaderView.ResizeMode.ResizeToContents
        )
        self.table.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.table.customContextMenuRequested.connect(self._show_context_menu)
        self.table.doubleClicked.connect(self._on_double_click)
        layout.addWidget(self.table, 1)

        # Action buttons
        actions_row = QHBoxLayout()
        self.add_btn = QPushButton(self.tr("+ Add Class"))
        self.add_btn.setStyleSheet(get_ok_btn_style())
        self.add_btn.clicked.connect(self._on_add_class)
        self.rename_btn = QPushButton(self.tr("Rename..."))
        self.rename_btn.setStyleSheet(get_cancel_btn_style())
        self.rename_btn.clicked.connect(self._on_rename)
        self.merge_btn = QPushButton(self.tr("Merge..."))
        self.merge_btn.setStyleSheet(get_cancel_btn_style())
        self.merge_btn.clicked.connect(self._on_merge)
        self.delete_btn = QPushButton(self.tr("Delete..."))
        self.delete_btn.setStyleSheet(get_cancel_btn_style())
        self.delete_btn.clicked.connect(self._on_delete)
        actions_row.addWidget(self.add_btn)
        actions_row.addWidget(self.rename_btn)
        actions_row.addWidget(self.merge_btn)
        actions_row.addWidget(self.delete_btn)
        actions_row.addStretch()
        layout.addLayout(actions_row)

        # Progress
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        # Bottom buttons
        btn_row = QHBoxLayout()
        btn_row.addStretch()
        self.close_btn = QPushButton(self.tr("Close"))
        self.close_btn.setStyleSheet(get_cancel_btn_style())
        self.close_btn.clicked.connect(self.accept)
        btn_row.addWidget(self.close_btn)
        layout.addLayout(btn_row)

    def _count_instances(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for path in self._annotation_paths:
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except (OSError, json.JSONDecodeError):
                continue
            for s in data.get("shapes", []):
                label = s.get("label")
                if label:
                    counts[label] = counts.get(label, 0) + 1
        return counts

    def _refresh(self) -> None:
        counts = self._count_instances()
        self.table.setRowCount(0)
        # Merge any labels that exist in annotations but not in the project class list
        known = {c["name"] for c in self._classes}
        for label in sorted(counts):
            if label not in known:
                self._classes.append({"name": label, "color": "#888888"})

        for cls in self._classes:
            row = self.table.rowCount()
            self.table.insertRow(row)
            name_item = QTableWidgetItem(cls["name"])
            name_item.setData(Qt.ItemDataRole.UserRole, cls["name"])
            self.table.setItem(row, 0, name_item)

            color_item = QTableWidgetItem(cls.get("color", "#888888"))
            try:
                color_item.setBackground(QColor(cls.get("color", "#888888")))
            except Exception:
                pass
            self.table.setItem(row, 1, color_item)

            cnt = counts.get(cls["name"], 0)
            cnt_item = QTableWidgetItem(str(cnt))
            cnt_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.table.setItem(row, 2, cnt_item)

    def _selected_class_names(self) -> List[str]:
        rows = sorted({r.row() for r in self.table.selectedIndexes()})
        names = []
        for r in rows:
            item = self.table.item(r, 0)
            if item:
                names.append(item.data(Qt.ItemDataRole.UserRole))
        return names

    def _show_context_menu(self, pos) -> None:
        menu = QMenu(self)
        menu.addAction(self.tr("Rename..."), self._on_rename)
        menu.addAction(self.tr("Merge..."), self._on_merge)
        menu.addSeparator()
        menu.addAction(self.tr("Delete..."), self._on_delete)
        menu.exec(self.table.viewport().mapToGlobal(pos))

    def _on_double_click(self, index) -> None:
        if index.column() == 1:
            self._pick_color_for_row(index.row())

    def _pick_color_for_row(self, row: int) -> None:
        if row < 0 or row >= len(self._classes):
            return
        current = QColor(self._classes[row].get("color", "#888888"))
        color = QColorDialog.getColor(current, self, self.tr("Choose color"))
        if color.isValid():
            self._classes[row]["color"] = color.name()
            self._dirty = True
            item = self.table.item(row, 1)
            if item:
                item.setText(color.name())
                item.setBackground(color)
            self.classes_changed.emit(self._classes)

    def _on_add_class(self) -> None:
        name, ok = QInputDialog.getText(
            self, self.tr("Add class"),
            self.tr("Class name:"),
        )
        if not ok or not name.strip():
            return
        name = name.strip()
        if any(c["name"] == name for c in self._classes):
            QMessageBox.warning(
                self, self.tr("Duplicate"),
                self.tr("A class with that name already exists."),
            )
            return
        self._classes.append({"name": name, "color": "#888888"})
        self._dirty = True
        self._refresh()
        self.classes_changed.emit(self._classes)

    def _on_rename(self) -> None:
        names = self._selected_class_names()
        if len(names) != 1:
            QMessageBox.information(
                self, self.tr("Select one class"),
                self.tr("Please select exactly one class to rename."),
            )
            return
        old = names[0]
        new, ok = QInputDialog.getText(
            self, self.tr("Rename class"),
            self.tr("New name for '%s':") % old,
            text=old,
        )
        if not ok or not new.strip() or new.strip() == old:
            return
        new = new.strip()
        if any(c["name"] == new for c in self._classes):
            QMessageBox.warning(
                self, self.tr("Duplicate"),
                self.tr(
                    "A class named '%s' already exists. "
                    "Use Merge instead."
                ) % new,
            )
            return

        reply = QMessageBox.question(
            self, self.tr("Confirm rename"),
            self.tr(
                "Rename class '%s' to '%s'?\n\n"
                "This will update all existing annotations "
                "(%d files will be scanned)."
            ) % (old, new, len(self._annotation_paths)),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        # Update local class list first
        for c in self._classes:
            if c["name"] == old:
                c["name"] = new
                break

        self._run_rename_thread("rename", [old], new)

    def _on_merge(self) -> None:
        names = self._selected_class_names()
        if len(names) < 2:
            QMessageBox.information(
                self, self.tr("Select multiple classes"),
                self.tr("Please select 2 or more classes to merge."),
            )
            return
        target, ok = QInputDialog.getItem(
            self, self.tr("Merge classes"),
            self.tr("Merge into which class?"),
            names, 0, False,
        )
        if not ok or not target:
            return
        sources = [n for n in names if n != target]
        if not sources:
            return

        reply = QMessageBox.question(
            self, self.tr("Confirm merge"),
            self.tr(
                "Merge %d classes (%s) into '%s'?\n\n"
                "This will update all annotations."
            ) % (len(sources), ", ".join(sources), target),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        # Update local class list: remove sources
        self._classes = [c for c in self._classes if c["name"] not in sources]

        self._run_rename_thread("merge", sources, target)

    def _on_delete(self) -> None:
        names = self._selected_class_names()
        if not names:
            return
        reply = QMessageBox.question(
            self, self.tr("Confirm delete"),
            self.tr(
                "Delete %d classes (%s) and ALL their annotation shapes?\n\n"
                "This cannot be undone. Annotation files will be modified in place."
            ) % (len(names), ", ".join(names)),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            defaultButton=QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        self._classes = [c for c in self._classes if c["name"] not in names]
        self._run_rename_thread("delete", names, None)

    def _run_rename_thread(
        self, action: str, from_labels: List[str], to_label: Optional[str]
    ) -> None:
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, len(self._annotation_paths))
        self.progress_bar.setValue(0)
        self.setEnabled(False)

        self._thread = ClassRenameThread(
            self._annotation_paths, action, from_labels, to_label
        )
        self._thread.progress.connect(self._on_progress)
        self._thread.finished.connect(self._on_thread_done)
        self._thread.start()

    def _on_progress(self, current: int, total: int) -> None:
        self.progress_bar.setValue(current)

    def _on_thread_done(
        self, success: bool, error: str, files_changed: int, shapes_changed: int
    ) -> None:
        self.progress_bar.setVisible(False)
        self.setEnabled(True)
        self._dirty = True
        self.classes_changed.emit(self._classes)
        self._refresh()

        if success:
            QMessageBox.information(
                self, self.tr("Done"),
                self.tr("Updated %d files (%d shapes).")
                % (files_changed, shapes_changed),
            )
        else:
            QMessageBox.critical(
                self, self.tr("Error"),
                self.tr("An error occurred: %s\n\nPartial update: %d files, %d shapes.")
                % (error, files_changed, shapes_changed),
            )

    def classes(self) -> List[Dict]:
        return list(self._classes)

    def is_dirty(self) -> bool:
        return self._dirty
