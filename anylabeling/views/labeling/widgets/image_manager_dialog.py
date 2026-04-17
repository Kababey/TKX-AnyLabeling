"""Image management dialog.

Lets users add images (single files or folders) to the current dataset
with resolution consistency checks and optional auto-resize, and
remove images from the project (without deleting original files).
"""

import os
import os.path as osp
import shutil
from typing import List, Optional

from PyQt6 import QtCore, QtWidgets
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QGroupBox,
    QHeaderView,
)

from anylabeling.views.labeling.logger import logger
from anylabeling.views.labeling.utils.image_resizer import (
    ResizeMode,
    batch_resize,
    check_resolution_consistency,
    detect_target_resolution,
    get_image_size,
    resize_image,
    transform_annotation,
)
from anylabeling.views.labeling.utils.style import (
    get_cancel_btn_style,
    get_dialog_style,
    get_ok_btn_style,
)


IMAGE_EXTENSIONS = (
    ".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff",
)


def _collect_images_from_folder(folder: str) -> List[str]:
    """Return sorted list of image files in folder (non-recursive)."""
    if not osp.isdir(folder):
        return []
    try:
        entries = os.listdir(folder)
    except OSError:
        return []
    out = []
    for e in entries:
        full = osp.join(folder, e)
        if osp.isfile(full) and e.lower().endswith(IMAGE_EXTENSIONS):
            out.append(full)
    return sorted(out)


class AddImagesThread(QThread):
    """Background worker that copies/resizes images into the dataset."""

    progress = pyqtSignal(int, int, str)  # current, total, message
    finished = pyqtSignal(bool, str, list)  # success, error, added_paths

    def __init__(
        self,
        source_images: List[str],
        target_dir: str,
        target_resolution: Optional[tuple],
        resize_mode: ResizeMode,
        copy_originals: bool = True,
    ):
        super().__init__()
        self.source_images = source_images
        self.target_dir = target_dir
        self.target_resolution = target_resolution
        self.resize_mode = resize_mode
        self.copy_originals = copy_originals
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    def run(self):
        added = []
        total = len(self.source_images)
        try:
            os.makedirs(self.target_dir, exist_ok=True)
            for i, src in enumerate(self.source_images, 1):
                if self._cancelled:
                    break
                name = osp.basename(src)
                dst = osp.join(self.target_dir, name)
                # Avoid overwriting existing file in dataset
                if osp.exists(dst) and osp.abspath(src) != osp.abspath(dst):
                    base, ext = osp.splitext(name)
                    j = 1
                    while osp.exists(dst):
                        dst = osp.join(
                            self.target_dir, f"{base}_{j}{ext}"
                        )
                        j += 1

                if (
                    self.target_resolution
                    and self.resize_mode != ResizeMode.NONE
                    and self.target_resolution != (0, 0)
                ):
                    result = resize_image(
                        src, dst, self.target_resolution, self.resize_mode
                    )
                    if not result.success:
                        self.progress.emit(
                            i, total, f"Failed: {name}: {result.error}"
                        )
                        continue
                else:
                    # Just copy
                    if osp.abspath(src) != osp.abspath(dst):
                        if self.copy_originals:
                            shutil.copy2(src, dst)
                        else:
                            # Hardlink fallback to copy
                            try:
                                os.link(src, dst)
                            except OSError:
                                shutil.copy2(src, dst)

                added.append(dst)
                self.progress.emit(i, total, f"Added: {name}")

            self.finished.emit(True, "", added)
        except Exception as exc:
            logger.error("AddImagesThread failed: %s", exc)
            self.finished.emit(False, str(exc), added)


class AddImagesDialog(QDialog):
    """Dialog to add images to the current dataset with resolution handling."""

    def __init__(
        self,
        target_dir: str,
        existing_images: List[str],
        suggested_resolution: Optional[tuple] = None,
        parent=None,
    ):
        super().__init__(parent)
        self._target_dir = target_dir
        self._existing_images = existing_images
        self._suggested_resolution = suggested_resolution
        self._selected_images: List[str] = []
        self._result_paths: List[str] = []
        self._thread: Optional[AddImagesThread] = None

        self.setWindowTitle(self.tr("Add Images"))
        self.setMinimumSize(720, 520)
        self.setStyleSheet(get_dialog_style())

        self._build_ui()
        self._refresh_preview()

    # -- UI --

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(12)

        # --- Source selection ---
        src_group = QGroupBox(self.tr("Source"))
        src_layout = QHBoxLayout(src_group)
        self.add_files_btn = QPushButton(self.tr("Add Files..."))
        self.add_files_btn.setStyleSheet(get_ok_btn_style())
        self.add_files_btn.clicked.connect(self._on_add_files)
        self.add_folder_btn = QPushButton(self.tr("Add Folder..."))
        self.add_folder_btn.setStyleSheet(get_ok_btn_style())
        self.add_folder_btn.clicked.connect(self._on_add_folder)
        self.clear_btn = QPushButton(self.tr("Clear"))
        self.clear_btn.setStyleSheet(get_cancel_btn_style())
        self.clear_btn.clicked.connect(self._on_clear)
        src_layout.addWidget(self.add_files_btn)
        src_layout.addWidget(self.add_folder_btn)
        src_layout.addStretch()
        src_layout.addWidget(self.clear_btn)
        layout.addWidget(src_group)

        # --- Preview table ---
        preview_group = QGroupBox(self.tr("Images to add"))
        preview_layout = QVBoxLayout(preview_group)
        self.summary_label = QLabel(self.tr("No images selected."))
        preview_layout.addWidget(self.summary_label)

        self.preview_table = QTableWidget(0, 3)
        self.preview_table.setHorizontalHeaderLabels([
            self.tr("Filename"),
            self.tr("Resolution"),
            self.tr("Status"),
        ])
        self.preview_table.setEditTriggers(
            QAbstractItemView.EditTrigger.NoEditTriggers
        )
        self.preview_table.setSelectionMode(
            QAbstractItemView.SelectionMode.ExtendedSelection
        )
        self.preview_table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeMode.Stretch
        )
        self.preview_table.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.ResizeMode.ResizeToContents
        )
        self.preview_table.horizontalHeader().setSectionResizeMode(
            2, QHeaderView.ResizeMode.ResizeToContents
        )
        preview_layout.addWidget(self.preview_table)
        layout.addWidget(preview_group, 1)

        # --- Resolution handling ---
        res_group = QGroupBox(self.tr("Resolution handling"))
        res_layout = QVBoxLayout(res_group)

        row1 = QHBoxLayout()
        self.resize_checkbox = QCheckBox(self.tr("Auto-resize mismatched images"))
        self.resize_checkbox.setChecked(True)
        self.resize_checkbox.stateChanged.connect(self._on_resize_toggled)
        row1.addWidget(self.resize_checkbox)
        row1.addStretch()
        res_layout.addLayout(row1)

        row2 = QHBoxLayout()
        row2.addWidget(QLabel(self.tr("Target width:")))
        self.target_w = QSpinBox()
        self.target_w.setRange(0, 16384)
        self.target_w.setValue(
            self._suggested_resolution[0] if self._suggested_resolution else 1024
        )
        self.target_w.valueChanged.connect(self._refresh_preview)
        row2.addWidget(self.target_w)

        row2.addWidget(QLabel(self.tr("Target height:")))
        self.target_h = QSpinBox()
        self.target_h.setRange(0, 16384)
        self.target_h.setValue(
            self._suggested_resolution[1] if self._suggested_resolution else 1024
        )
        self.target_h.valueChanged.connect(self._refresh_preview)
        row2.addWidget(self.target_h)

        row2.addWidget(QLabel(self.tr("Mode:")))
        self.mode_combo = QComboBox()
        self.mode_combo.addItem(
            self.tr("Letterbox (keep aspect, pad)"), ResizeMode.LETTERBOX.value
        )
        self.mode_combo.addItem(
            self.tr("Center crop (keep aspect, crop)"), ResizeMode.CENTER_CROP.value
        )
        self.mode_combo.addItem(
            self.tr("Stretch (distort)"), ResizeMode.STRETCH.value
        )
        row2.addWidget(self.mode_combo)
        row2.addStretch()
        res_layout.addLayout(row2)
        layout.addWidget(res_group)

        # --- Progress ---
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        self.status_label = QLabel("")
        self.status_label.setStyleSheet("color: #888;")
        layout.addWidget(self.status_label)

        # --- Buttons ---
        btn_row = QHBoxLayout()
        btn_row.addStretch()
        self.cancel_btn = QPushButton(self.tr("Cancel"))
        self.cancel_btn.setStyleSheet(get_cancel_btn_style())
        self.cancel_btn.clicked.connect(self.reject)
        self.add_btn = QPushButton(self.tr("Add to Dataset"))
        self.add_btn.setStyleSheet(get_ok_btn_style())
        self.add_btn.clicked.connect(self._on_confirm)
        self.add_btn.setEnabled(False)
        btn_row.addWidget(self.cancel_btn)
        btn_row.addWidget(self.add_btn)
        layout.addLayout(btn_row)

    # -- Handlers --

    def _on_add_files(self) -> None:
        filter_str = "Images (*.jpg *.jpeg *.png *.bmp *.webp *.tif *.tiff)"
        files, _ = QFileDialog.getOpenFileNames(
            self,
            self.tr("Select images to add"),
            "",
            filter_str,
        )
        if files:
            self._add_to_selection(files)

    def _on_add_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(
            self,
            self.tr("Select folder of images"),
            "",
            QFileDialog.Option.DontUseNativeDialog,
        )
        if folder:
            images = _collect_images_from_folder(folder)
            if not images:
                QMessageBox.information(
                    self,
                    self.tr("No images found"),
                    self.tr("The selected folder contains no supported images."),
                )
                return
            self._add_to_selection(images)

    def _on_clear(self) -> None:
        self._selected_images = []
        self._refresh_preview()

    def _add_to_selection(self, files: List[str]) -> None:
        seen = set(self._selected_images)
        for f in files:
            if f not in seen:
                self._selected_images.append(f)
                seen.add(f)
        self._refresh_preview()

    def _on_resize_toggled(self, state: int) -> None:
        enabled = state == Qt.CheckState.Checked.value
        self.target_w.setEnabled(enabled)
        self.target_h.setEnabled(enabled)
        self.mode_combo.setEnabled(enabled)
        self._refresh_preview()

    def _refresh_preview(self) -> None:
        self.preview_table.setRowCount(0)
        target = (self.target_w.value(), self.target_h.value())
        will_resize = (
            self.resize_checkbox.isChecked()
            and target[0] > 0
            and target[1] > 0
        )

        match_count = 0
        mismatch_count = 0
        unreadable_count = 0

        for path in self._selected_images:
            size = get_image_size(path)
            row = self.preview_table.rowCount()
            self.preview_table.insertRow(row)
            name_item = QTableWidgetItem(osp.basename(path))
            name_item.setToolTip(path)
            self.preview_table.setItem(row, 0, name_item)

            if size is None:
                self.preview_table.setItem(
                    row, 1, QTableWidgetItem("?")
                )
                status_item = QTableWidgetItem(self.tr("Unreadable"))
                status_item.setForeground(QColor("#d32f2f"))
                self.preview_table.setItem(row, 2, status_item)
                unreadable_count += 1
                continue

            self.preview_table.setItem(
                row, 1, QTableWidgetItem(f"{size[0]}x{size[1]}")
            )
            if size == target:
                status = self.tr("Match")
                color = QColor("#2e7d32")
                match_count += 1
            elif will_resize:
                status = self.tr("Will be resized")
                color = QColor("#ed6c02")
                mismatch_count += 1
            else:
                status = self.tr("Mismatch (will not be resized)")
                color = QColor("#d32f2f")
                mismatch_count += 1
            status_item = QTableWidgetItem(status)
            status_item.setForeground(color)
            self.preview_table.setItem(row, 2, status_item)

        total = len(self._selected_images)
        if total == 0:
            self.summary_label.setText(self.tr("No images selected."))
            self.add_btn.setEnabled(False)
        else:
            parts = [
                self.tr("%d total") % total,
                self.tr("%d match target") % match_count,
                self.tr("%d mismatch") % mismatch_count,
            ]
            if unreadable_count:
                parts.append(self.tr("%d unreadable") % unreadable_count)
            self.summary_label.setText("  •  ".join(parts))
            self.add_btn.setEnabled(True)

    def _on_confirm(self) -> None:
        target = (self.target_w.value(), self.target_h.value())
        will_resize = (
            self.resize_checkbox.isChecked()
            and target[0] > 0
            and target[1] > 0
        )
        mode = ResizeMode(self.mode_combo.currentData()) if will_resize else ResizeMode.NONE
        effective_target = target if will_resize else None

        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, len(self._selected_images))
        self.progress_bar.setValue(0)
        self.add_btn.setEnabled(False)
        self.cancel_btn.setEnabled(False)

        self._thread = AddImagesThread(
            source_images=self._selected_images,
            target_dir=self._target_dir,
            target_resolution=effective_target,
            resize_mode=mode,
        )
        self._thread.progress.connect(self._on_progress)
        self._thread.finished.connect(self._on_thread_finished)
        self._thread.start()

    def _on_progress(self, current: int, total: int, message: str) -> None:
        self.progress_bar.setValue(current)
        self.status_label.setText(message)

    def _on_thread_finished(
        self, success: bool, error: str, added: List[str]
    ) -> None:
        self._result_paths = added
        if success:
            QMessageBox.information(
                self,
                self.tr("Images added"),
                self.tr("Successfully added %d images.") % len(added),
            )
            self.accept()
        else:
            QMessageBox.warning(
                self,
                self.tr("Error adding images"),
                self.tr("Error: %s\n%d images were added before the failure.")
                % (error, len(added)),
            )
            self.reject()

    def result_paths(self) -> List[str]:
        return list(self._result_paths)

    def reject(self) -> None:
        if self._thread and self._thread.isRunning():
            self._thread.cancel()
            self._thread.wait(3000)
        super().reject()
