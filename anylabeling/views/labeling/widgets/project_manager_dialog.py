"""Project manager dialog - the Roboflow-style home screen for
creating, opening, and managing multiple datasets as projects.
"""

import os
import os.path as osp
from typing import List, Optional

from PyQt6 import QtCore, QtWidgets
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QGroupBox,
    QHeaderView,
    QSplitter,
    QWidget,
)

from anylabeling.views.labeling.logger import logger
from anylabeling.views.labeling.utils.project_manager import (
    PROJECT_FILE_NAME,
    ProjectInfo,
    ProjectManager,
)
from anylabeling.views.labeling.utils.style import (
    get_cancel_btn_style,
    get_dialog_style,
    get_ok_btn_style,
)


class NewProjectDialog(QDialog):
    """Dialog for creating a new project."""

    def __init__(self, project_manager: ProjectManager, parent=None):
        super().__init__(parent)
        self._pm = project_manager
        self._created_info: Optional[ProjectInfo] = None

        self.setWindowTitle(self.tr("New Project"))
        self.setMinimumWidth(540)
        self.setStyleSheet(get_dialog_style())

        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(12)

        # Name
        layout.addWidget(QLabel(self.tr("Project name *")))
        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText(self.tr("My Dataset"))
        layout.addWidget(self.name_edit)

        # Description
        layout.addWidget(QLabel(self.tr("Description (optional)")))
        self.desc_edit = QPlainTextEdit()
        self.desc_edit.setMaximumHeight(70)
        layout.addWidget(self.desc_edit)

        # Directory
        layout.addWidget(QLabel(self.tr("Project directory *")))
        dir_row = QHBoxLayout()
        self.dir_edit = QLineEdit()
        self.dir_edit.setPlaceholderText(
            self.tr("Select an empty or new directory")
        )
        self.browse_btn = QPushButton(self.tr("Browse..."))
        self.browse_btn.setStyleSheet(get_cancel_btn_style())
        self.browse_btn.clicked.connect(self._on_browse)
        dir_row.addWidget(self.dir_edit)
        dir_row.addWidget(self.browse_btn)
        layout.addLayout(dir_row)

        # Classes
        layout.addWidget(QLabel(
            self.tr("Classes (comma-separated, optional)")
        ))
        self.classes_edit = QLineEdit()
        self.classes_edit.setPlaceholderText(
            self.tr("car, person, dog")
        )
        layout.addWidget(self.classes_edit)

        # Target resolution
        res_group = QGroupBox(self.tr("Target resolution (optional)"))
        res_layout = QHBoxLayout(res_group)
        res_layout.addWidget(QLabel(self.tr("Width:")))
        self.w_edit = QSpinBox()
        self.w_edit.setRange(0, 16384)
        self.w_edit.setValue(0)
        self.w_edit.setSpecialValueText(self.tr("Not set"))
        res_layout.addWidget(self.w_edit)
        res_layout.addWidget(QLabel(self.tr("Height:")))
        self.h_edit = QSpinBox()
        self.h_edit.setRange(0, 16384)
        self.h_edit.setValue(0)
        self.h_edit.setSpecialValueText(self.tr("Not set"))
        res_layout.addWidget(self.h_edit)
        res_layout.addStretch()
        layout.addWidget(res_group)

        # Buttons
        btn_row = QHBoxLayout()
        btn_row.addStretch()
        self.cancel_btn = QPushButton(self.tr("Cancel"))
        self.cancel_btn.setStyleSheet(get_cancel_btn_style())
        self.cancel_btn.clicked.connect(self.reject)
        self.create_btn = QPushButton(self.tr("Create Project"))
        self.create_btn.setStyleSheet(get_ok_btn_style())
        self.create_btn.clicked.connect(self._on_create)
        btn_row.addWidget(self.cancel_btn)
        btn_row.addWidget(self.create_btn)
        layout.addLayout(btn_row)

    def _on_browse(self) -> None:
        folder = QFileDialog.getExistingDirectory(
            self,
            self.tr("Select project directory"),
            osp.expanduser("~"),
            QFileDialog.Option.DontUseNativeDialog,
        )
        if folder:
            self.dir_edit.setText(folder)
            # Auto-fill name if empty
            if not self.name_edit.text().strip():
                self.name_edit.setText(osp.basename(folder))

    def _on_create(self) -> None:
        name = self.name_edit.text().strip()
        proj_dir = self.dir_edit.text().strip()
        if not name:
            QMessageBox.warning(
                self,
                self.tr("Missing name"),
                self.tr("Please enter a project name."),
            )
            return
        if not proj_dir:
            QMessageBox.warning(
                self,
                self.tr("Missing directory"),
                self.tr("Please select a project directory."),
            )
            return
        if not osp.isdir(proj_dir):
            try:
                os.makedirs(proj_dir, exist_ok=True)
            except OSError as exc:
                QMessageBox.critical(
                    self,
                    self.tr("Cannot create directory"),
                    self.tr("Could not create directory: %s") % exc,
                )
                return

        if self._pm.is_project_dir(proj_dir):
            QMessageBox.warning(
                self,
                self.tr("Directory is already a project"),
                self.tr("That directory already contains a project. "
                        "Open it instead."),
            )
            return

        classes_raw = self.classes_edit.text().strip()
        classes = [
            c.strip() for c in classes_raw.split(",") if c.strip()
        ] if classes_raw else []

        settings = {}
        tw, th = self.w_edit.value(), self.h_edit.value()
        if tw > 0 and th > 0:
            settings["target_resolution"] = [tw, th]
            settings["auto_resize_new_images"] = True

        try:
            info = self._pm.create_project(
                project_dir=proj_dir,
                name=name,
                description=self.desc_edit.toPlainText().strip(),
                classes=classes,
                settings=settings,
            )
            self._created_info = info
            self.accept()
        except ValueError as exc:
            QMessageBox.critical(
                self,
                self.tr("Cannot create project"),
                str(exc),
            )

    def created_project(self) -> Optional[ProjectInfo]:
        return self._created_info


class ProjectManagerDialog(QDialog):
    """Home-screen dialog listing recent projects and letting user
    create/open/remove projects.

    Signals:
        project_opened: emitted with project path when the user opens a project.
    """

    project_opened = pyqtSignal(str)

    def __init__(self, project_manager: ProjectManager, parent=None):
        super().__init__(parent)
        self._pm = project_manager
        self._selected_path: Optional[str] = None

        self.setWindowTitle(self.tr("X-AnyLabeling Projects"))
        self.setMinimumSize(820, 520)
        self.setStyleSheet(get_dialog_style())

        self._build_ui()
        self._refresh()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(12)

        header = QLabel(self.tr("Your Projects"))
        header.setStyleSheet("font-size: 18px; font-weight: 600;")
        layout.addWidget(header)

        # Action buttons at top
        actions_row = QHBoxLayout()
        self.new_btn = QPushButton(self.tr("+ New Project"))
        self.new_btn.setStyleSheet(get_ok_btn_style())
        self.new_btn.clicked.connect(self._on_new_project)
        self.open_dir_btn = QPushButton(self.tr("Open Existing..."))
        self.open_dir_btn.setStyleSheet(get_cancel_btn_style())
        self.open_dir_btn.clicked.connect(self._on_open_existing)
        actions_row.addWidget(self.new_btn)
        actions_row.addWidget(self.open_dir_btn)
        actions_row.addStretch()
        self.refresh_btn = QPushButton(self.tr("Refresh"))
        self.refresh_btn.setStyleSheet(get_cancel_btn_style())
        self.refresh_btn.clicked.connect(self._refresh)
        actions_row.addWidget(self.refresh_btn)
        layout.addLayout(actions_row)

        # Recent projects table
        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels([
            self.tr("Name"),
            self.tr("Path"),
            self.tr("Images"),
            self.tr("Classes"),
            self.tr("Last opened"),
        ])
        self.table.setEditTriggers(
            QAbstractItemView.EditTrigger.NoEditTriggers
        )
        self.table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows
        )
        self.table.setSelectionMode(
            QAbstractItemView.SelectionMode.SingleSelection
        )
        self.table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeMode.ResizeToContents
        )
        self.table.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.ResizeMode.Stretch
        )
        self.table.horizontalHeader().setSectionResizeMode(
            2, QHeaderView.ResizeMode.ResizeToContents
        )
        self.table.horizontalHeader().setSectionResizeMode(
            3, QHeaderView.ResizeMode.ResizeToContents
        )
        self.table.horizontalHeader().setSectionResizeMode(
            4, QHeaderView.ResizeMode.ResizeToContents
        )
        self.table.doubleClicked.connect(self._on_open_selected)
        layout.addWidget(self.table, 1)

        # Bottom button row
        btn_row = QHBoxLayout()
        self.remove_btn = QPushButton(self.tr("Remove from Recent"))
        self.remove_btn.setStyleSheet(get_cancel_btn_style())
        self.remove_btn.clicked.connect(self._on_remove_recent)
        self.remove_btn.setToolTip(
            self.tr("Remove from the recent list (does NOT delete any files)")
        )
        btn_row.addWidget(self.remove_btn)
        btn_row.addStretch()
        self.close_btn = QPushButton(self.tr("Close"))
        self.close_btn.setStyleSheet(get_cancel_btn_style())
        self.close_btn.clicked.connect(self.reject)
        self.open_btn = QPushButton(self.tr("Open Selected"))
        self.open_btn.setStyleSheet(get_ok_btn_style())
        self.open_btn.clicked.connect(self._on_open_selected)
        btn_row.addWidget(self.close_btn)
        btn_row.addWidget(self.open_btn)
        layout.addLayout(btn_row)

    def _refresh(self) -> None:
        self.table.setRowCount(0)
        entries = self._pm.list_recent_projects()
        for entry in entries:
            path = entry.get("path", "")
            name = entry.get("name", osp.basename(path) or "?")
            last = entry.get("last_opened", "")
            exists = entry.get("exists", False)

            # Try to load current stats / class count
            image_count = "?"
            class_count = "?"
            if exists:
                try:
                    info = self._pm.load_project(path)
                    stats = info.stats or {}
                    image_count = str(stats.get("image_count", "?"))
                    class_count = str(len(info.classes or []))
                except ValueError:
                    exists = False

            row = self.table.rowCount()
            self.table.insertRow(row)
            name_item = QTableWidgetItem(name)
            name_item.setData(Qt.ItemDataRole.UserRole, path)
            if not exists:
                name_item.setForeground(Qt.GlobalColor.gray)
                name_item.setToolTip(self.tr("Project file missing at this path"))
            self.table.setItem(row, 0, name_item)
            self.table.setItem(row, 1, QTableWidgetItem(path))
            self.table.setItem(row, 2, QTableWidgetItem(image_count))
            self.table.setItem(row, 3, QTableWidgetItem(class_count))
            self.table.setItem(row, 4, QTableWidgetItem(last))

        if entries:
            self.table.selectRow(0)

    def _selected_row_path(self) -> Optional[str]:
        row = self.table.currentRow()
        if row < 0:
            return None
        item = self.table.item(row, 0)
        if item is None:
            return None
        return item.data(Qt.ItemDataRole.UserRole)

    def _on_new_project(self) -> None:
        dlg = NewProjectDialog(self._pm, parent=self)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            info = dlg.created_project()
            if info:
                self._selected_path = info.path
                self.project_opened.emit(info.path)
                self.accept()

    def _on_open_existing(self) -> None:
        folder = QFileDialog.getExistingDirectory(
            self,
            self.tr("Select existing project directory"),
            osp.expanduser("~"),
            QFileDialog.Option.DontUseNativeDialog,
        )
        if not folder:
            return
        if self._pm.is_project_dir(folder):
            self._open_path(folder)
        else:
            # Offer to adopt folder as a new project
            reply = QMessageBox.question(
                self,
                self.tr("Not a project"),
                self.tr(
                    "That directory is not a project. "
                    "Would you like to create a project from it?"
                ),
                QMessageBox.StandardButton.Yes
                | QMessageBox.StandardButton.No,
            )
            if reply == QMessageBox.StandardButton.Yes:
                try:
                    info = self._pm.create_project(
                        folder, name=osp.basename(folder)
                    )
                    self._selected_path = info.path
                    self.project_opened.emit(info.path)
                    self.accept()
                except ValueError as exc:
                    QMessageBox.critical(
                        self,
                        self.tr("Cannot create project"),
                        str(exc),
                    )

    def _on_open_selected(self) -> None:
        path = self._selected_row_path()
        if not path:
            return
        self._open_path(path)

    def _open_path(self, path: str) -> None:
        if not self._pm.is_project_dir(path):
            QMessageBox.warning(
                self,
                self.tr("Project not found"),
                self.tr("The project file no longer exists at:\n%s") % path,
            )
            return
        try:
            info = self._pm.open_project(path)
            self._selected_path = info.path
            self.project_opened.emit(info.path)
            self.accept()
        except ValueError as exc:
            QMessageBox.critical(
                self,
                self.tr("Cannot open project"),
                str(exc),
            )

    def _on_remove_recent(self) -> None:
        path = self._selected_row_path()
        if not path:
            return
        reply = QMessageBox.question(
            self,
            self.tr("Remove from recent"),
            self.tr(
                "Remove '%s' from the recent projects list?\n\n"
                "This will NOT delete any files."
            ) % path,
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            self._pm.remove_from_recent(path)
            self._refresh()

    def selected_path(self) -> Optional[str]:
        return self._selected_path
