"""Dockable queue widget listing unlabeled images."""

import os.path as osp

from PyQt6 import QtCore, QtWidgets
from PyQt6.QtCore import Qt, pyqtSignal

from anylabeling.views.labeling.utils.annotation_status import scan_unlabeled


_MAX_ROWS = 2000


class AnnotationQueueDock(QtWidgets.QDockWidget):
    """Roboflow-style queue of images lacking annotations."""

    image_selected = pyqtSignal(str)
    mark_reviewed_requested = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("AnnotateQueueDock")
        self.setWindowTitle(self.tr("Annotate Queue"))

        container = QtWidgets.QWidget(self)
        layout = QtWidgets.QVBoxLayout(container)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(4)

        top_row = QtWidgets.QHBoxLayout()
        top_row.setSpacing(6)
        self._refresh_btn = QtWidgets.QPushButton(self.tr("Refresh"))
        self._refresh_btn.clicked.connect(self._emit_refresh)
        self._count_label = QtWidgets.QLabel("")
        self._hide_reviewed_chk = QtWidgets.QCheckBox(
            self.tr("Hide reviewed-empty")
        )
        self._hide_reviewed_chk.setChecked(True)
        self._hide_reviewed_chk.stateChanged.connect(
            lambda _s: self._rerender()
        )
        top_row.addWidget(self._refresh_btn)
        top_row.addWidget(self._count_label, 1)
        top_row.addWidget(self._hide_reviewed_chk)
        layout.addLayout(top_row)

        self._list = QtWidgets.QListWidget(container)
        self._list.setSelectionMode(
            QtWidgets.QAbstractItemView.SelectionMode.SingleSelection
        )
        self._list.itemDoubleClicked.connect(self._on_double_click)
        self._list.setContextMenuPolicy(
            Qt.ContextMenuPolicy.CustomContextMenu
        )
        self._list.customContextMenuRequested.connect(self._on_context_menu)
        layout.addWidget(self._list, 1)

        self.setWidget(container)

        self._last_paths = []
        self._annotations_dir = None
        self._refresh_callback = None
        self.hide()

    def set_refresh_callback(self, fn):
        self._refresh_callback = fn

    def _emit_refresh(self):
        if callable(self._refresh_callback):
            self._refresh_callback()

    def refresh(self, image_list, annotations_dir):
        """Re-scan image_list against annotations_dir and update the list."""
        self._annotations_dir = annotations_dir
        unlabeled = scan_unlabeled(image_list or [], annotations_dir or "")
        unlabeled = sorted(unlabeled, key=lambda p: osp.basename(p).lower())
        self._last_paths = unlabeled
        self._rerender()

    def _rerender(self):
        self._list.clear()
        paths = list(self._last_paths)
        total = len(paths)
        display = paths[:_MAX_ROWS]
        for p in display:
            item = QtWidgets.QListWidgetItem(osp.basename(p))
            item.setToolTip(p)
            item.setData(Qt.ItemDataRole.UserRole, p)
            self._list.addItem(item)
        if total > _MAX_ROWS:
            footer = QtWidgets.QListWidgetItem(
                self.tr("...and %d more") % (total - _MAX_ROWS)
            )
            footer.setFlags(Qt.ItemFlag.NoItemFlags)
            self._list.addItem(footer)
        self._count_label.setText(
            self.tr("%d unlabeled") % total
        )

    def remove_path(self, path):
        """Remove an image path from the queue (e.g. after it was annotated)."""
        if not path:
            return
        self._last_paths = [p for p in self._last_paths if p != path]
        for i in range(self._list.count()):
            item = self._list.item(i)
            stored = item.data(Qt.ItemDataRole.UserRole)
            if stored == path:
                self._list.takeItem(i)
                break
        self._count_label.setText(
            self.tr("%d unlabeled") % len(self._last_paths)
        )

    def set_current(self, path):
        """Highlight the row matching the given image path, if present."""
        if not path:
            return
        for i in range(self._list.count()):
            item = self._list.item(i)
            if item.data(Qt.ItemDataRole.UserRole) == path:
                self._list.setCurrentRow(i)
                return

    def _on_double_click(self, item):
        path = item.data(Qt.ItemDataRole.UserRole)
        if path:
            self.image_selected.emit(path)

    def keyPressEvent(self, event):
        if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            item = self._list.currentItem()
            if item is not None:
                path = item.data(Qt.ItemDataRole.UserRole)
                if path:
                    self.image_selected.emit(path)
                    return
        super().keyPressEvent(event)

    def _on_context_menu(self, pos):
        item = self._list.itemAt(pos)
        if item is None:
            return
        path = item.data(Qt.ItemDataRole.UserRole)
        if not path:
            return
        menu = QtWidgets.QMenu(self)
        act_open = menu.addAction(self.tr("Open image"))
        act_mark = menu.addAction(self.tr("Mark as reviewed (no objects)"))
        chosen = menu.exec(self._list.viewport().mapToGlobal(pos))
        if chosen is act_open:
            self.image_selected.emit(path)
        elif chosen is act_mark:
            self.mark_reviewed_requested.emit(path)
