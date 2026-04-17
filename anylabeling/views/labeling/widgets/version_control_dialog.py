"""GUI dialogs for annotation version control.

Provides dialogs for creating, browsing, comparing, and restoring
annotation version snapshots.  Built on top of the pure-logic
:class:`~anylabeling.views.labeling.utils.version_control.VersionManager`.
"""

from datetime import datetime

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
)

from anylabeling.views.labeling.logger import logger
from anylabeling.views.labeling.utils.style import (
    get_cancel_btn_style,
    get_dialog_style,
    get_highlight_button_style,
    get_normal_button_style,
    get_ok_btn_style,
)
from anylabeling.views.labeling.utils.theme import get_theme


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _format_size(size_bytes):
    """Format a byte count into a human-readable string."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    else:
        return f"{size_bytes / (1024 * 1024):.1f} MB"


def _format_timestamp(iso_timestamp):
    """Format an ISO timestamp into a user-friendly display string."""
    try:
        dt = datetime.fromisoformat(iso_timestamp)
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except (ValueError, TypeError):
        return str(iso_timestamp)


def _get_delete_btn_style():
    """Return a themed stylesheet for destructive / delete buttons."""
    t = get_theme()
    return f"""
        QPushButton {{
            background-color: {t["error"]};
            color: white;
            border: none;
            border-radius: 8px;
            font-weight: 500;
            min-width: 100px;
            height: 36px;
            padding: 0 12px;
        }}
        QPushButton:hover {{
            background-color: {t["error"]};
            opacity: 0.85;
        }}
        QPushButton:pressed {{
            background-color: {t["error"]};
            opacity: 0.7;
        }}
        QPushButton:disabled {{
            background-color: {t["surface"]};
            color: {t["text_secondary"]};
            border: 1px solid {t["border"]};
        }}
    """


def _get_diff_text_style():
    """Return a themed stylesheet for the diff detail text area."""
    t = get_theme()
    return f"""
        QTextEdit {{
            background-color: {t["background"]};
            color: {t["text"]};
            border: 1px solid {t["border"]};
            border-radius: 6px;
            padding: 8px;
            font-family: "Consolas", "Menlo", "Monaco", monospace;
            font-size: 13px;
        }}
    """


# ---------------------------------------------------------------------------
# CreateVersionDialog
# ---------------------------------------------------------------------------


class CreateVersionDialog(QDialog):
    """Simple dialog for entering a version name and description."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle(self.tr("Create Version"))
        self.setMinimumWidth(420)
        self.setModal(True)
        self.setWindowFlags(
            self.windowFlags()
            & ~Qt.WindowType.WindowContextHelpButtonHint
        )
        self.setStyleSheet(get_dialog_style())

        layout = QVBoxLayout()
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(16)

        # --- Name field ---
        name_label = QLabel(self.tr("Version Name"))
        name_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        layout.addWidget(name_label)

        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText(
            self.tr("e.g. baseline, after-review, final")
        )
        layout.addWidget(self.name_edit)

        # --- Description field ---
        desc_label = QLabel(self.tr("Description (optional)"))
        desc_label.setStyleSheet("font-size: 13px;")
        layout.addWidget(desc_label)

        self.desc_edit = QPlainTextEdit()
        self.desc_edit.setPlaceholderText(
            self.tr("Briefly describe what changed in this version...")
        )
        self.desc_edit.setMaximumHeight(100)
        layout.addWidget(self.desc_edit)

        layout.addStretch()

        # --- Buttons ---
        btn_layout = QHBoxLayout()
        btn_layout.setContentsMargins(0, 8, 0, 0)
        btn_layout.setSpacing(8)
        btn_layout.addStretch()

        cancel_btn = QPushButton(self.tr("Cancel"))
        cancel_btn.setStyleSheet(get_cancel_btn_style())
        cancel_btn.clicked.connect(self.reject)
        btn_layout.addWidget(cancel_btn)

        self.ok_btn = QPushButton(self.tr("OK"))
        self.ok_btn.setStyleSheet(get_ok_btn_style())
        self.ok_btn.clicked.connect(self._accept_if_valid)
        self.ok_btn.setDefault(True)
        btn_layout.addWidget(self.ok_btn)

        layout.addLayout(btn_layout)
        self.setLayout(layout)

    # -- accessors --

    def get_name(self):
        return self.name_edit.text().strip()

    def get_description(self):
        return self.desc_edit.toPlainText().strip()

    # -- internal --

    def _accept_if_valid(self):
        if not self.get_name():
            QMessageBox.warning(
                self,
                self.tr("Validation Error"),
                self.tr("Version name is required."),
            )
            self.name_edit.setFocus()
            return
        self.accept()


# ---------------------------------------------------------------------------
# VersionDiffDialog
# ---------------------------------------------------------------------------


class VersionDiffDialog(QDialog):
    """Shows differences between two version snapshots."""

    def __init__(self, version_diff, version_a_info, version_b_info,
                 parent=None):
        """
        Parameters
        ----------
        version_diff : VersionDiff
            The diff result from VersionManager.compare_versions().
        version_a_info : VersionInfo
            Metadata for the baseline version.
        version_b_info : VersionInfo
            Metadata for the comparison version.
        parent : QWidget, optional
        """
        super().__init__(parent)
        self._diff = version_diff
        self._a_info = version_a_info
        self._b_info = version_b_info

        self.setWindowTitle(self.tr("Version Comparison"))
        self.setMinimumSize(780, 520)
        self.setModal(True)
        self.setWindowFlags(
            self.windowFlags()
            & ~Qt.WindowType.WindowContextHelpButtonHint
        )
        self.setStyleSheet(get_dialog_style())

        layout = QVBoxLayout()
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(12)

        # --- Header ---
        header = QLabel(
            self.tr("Comparing: %s  →  %s")
            % (self._a_info.name, self._b_info.name)
        )
        header.setStyleSheet("font-weight: bold; font-size: 15px;")
        layout.addWidget(header)

        # --- Summary stats ---
        summary = self._diff.summary
        changed = (
            summary.get("added_images", 0)
            + summary.get("removed_images", 0)
            + summary.get("modified_images", 0)
        )
        summary_text = self.tr(
            "%n image(s) changed: %1 added, %2 removed, %3 modified",
            "",
            changed,
        ).replace("%1", str(summary.get("added_images", 0))) \
         .replace("%2", str(summary.get("removed_images", 0))) \
         .replace("%3", str(summary.get("modified_images", 0)))

        summary_label = QLabel(summary_text)
        summary_label.setStyleSheet("font-size: 13px;")
        layout.addWidget(summary_label)

        # --- Splitter: image list | detail panel ---
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left: changed images list
        self.image_list = QListWidget()
        self.image_list.setMinimumWidth(220)
        self.image_list.currentItemChanged.connect(self._on_image_selected)

        t = get_theme()
        color_map = {
            "added": t.get("primary", "#34C759"),
            "removed": t.get("error", "#FF453A"),
            "modified": t.get("warning", "#FF9F0A"),
        }

        for img_diff in self._diff.image_diffs:
            if img_diff.status == "unchanged":
                continue
            item = QListWidgetItem(img_diff.image_name)
            status_color = color_map.get(img_diff.status, t["text"])
            item.setForeground(QColor(status_color))

            status_label = {
                "added": self.tr("[Added]"),
                "removed": self.tr("[Removed]"),
                "modified": self.tr("[Modified]"),
            }.get(img_diff.status, "")
            item.setText(f"{status_label}  {img_diff.image_name}")
            item.setData(Qt.ItemDataRole.UserRole, img_diff)
            self.image_list.addItem(item)

        if self.image_list.count() == 0:
            placeholder = QListWidgetItem(self.tr("No differences found"))
            placeholder.setFlags(
                placeholder.flags() & ~Qt.ItemFlag.ItemIsSelectable
            )
            self.image_list.addItem(placeholder)

        splitter.addWidget(self.image_list)

        # Right: diff detail
        self.detail_view = QTextEdit()
        self.detail_view.setReadOnly(True)
        self.detail_view.setStyleSheet(_get_diff_text_style())
        self.detail_view.setMinimumWidth(340)
        self.detail_view.setPlaceholderText(
            self.tr("Select an image from the list to view details")
        )
        splitter.addWidget(self.detail_view)

        splitter.setSizes([280, 480])
        layout.addWidget(splitter, 1)

        # --- Close button ---
        btn_layout = QHBoxLayout()
        btn_layout.setContentsMargins(0, 8, 0, 0)
        btn_layout.addStretch()

        close_btn = QPushButton(self.tr("Close"))
        close_btn.setStyleSheet(get_cancel_btn_style())
        close_btn.clicked.connect(self.accept)
        btn_layout.addWidget(close_btn)

        layout.addLayout(btn_layout)
        self.setLayout(layout)

        # Auto-select the first item
        if self.image_list.count() > 0:
            first = self.image_list.item(0)
            if first.flags() & Qt.ItemFlag.ItemIsSelectable:
                self.image_list.setCurrentItem(first)

    # -- slots --

    def _on_image_selected(self, current, _previous):
        """Populate the detail panel when an image is selected."""
        if current is None:
            self.detail_view.clear()
            return

        img_diff = current.data(Qt.ItemDataRole.UserRole)
        if img_diff is None:
            self.detail_view.clear()
            return

        t = get_theme()
        added_color = t.get("primary", "#34C759")
        removed_color = t.get("error", "#FF453A")
        modified_color = t.get("warning", "#FF9F0A")
        text_color = t.get("text", "#FFFFFF")

        status_colors = {
            "added": added_color,
            "removed": removed_color,
            "modified": modified_color,
        }
        status_color = status_colors.get(img_diff.status, text_color)

        html_parts = []
        html_parts.append(
            f'<h3 style="color: {text_color};">{img_diff.image_name}</h3>'
        )
        html_parts.append(
            f'<p style="color: {status_color}; font-weight: bold;">'
            f'{self.tr("Status")}: {img_diff.status.capitalize()}</p>'
        )

        # Shape change summary
        shape_summary_parts = []
        if img_diff.added_shapes > 0:
            shape_summary_parts.append(
                f'<span style="color: {added_color};">'
                f'+{img_diff.added_shapes} {self.tr("added")}</span>'
            )
        if img_diff.removed_shapes > 0:
            shape_summary_parts.append(
                f'<span style="color: {removed_color};">'
                f'-{img_diff.removed_shapes} {self.tr("removed")}</span>'
            )
        if img_diff.modified_shapes > 0:
            shape_summary_parts.append(
                f'<span style="color: {modified_color};">'
                f'~{img_diff.modified_shapes} {self.tr("modified")}</span>'
            )

        if shape_summary_parts:
            html_parts.append(
                f'<p style="color: {text_color};">'
                f'{self.tr("Shapes")}: '
                + " &nbsp; ".join(shape_summary_parts)
                + "</p>"
            )

        # Per-shape details
        if img_diff.details:
            html_parts.append(f'<hr style="border-color: {t["border"]};">')
            html_parts.append(
                f'<p style="color: {text_color}; font-weight: bold;">'
                f'{self.tr("Shape Details")}:</p>'
            )
            html_parts.append("<ul>")
            for detail in img_diff.details:
                change_type = detail.get("change_type", "")
                label = detail.get("label", self.tr("unknown"))
                shape_type = detail.get("shape_type", "")
                color = status_colors.get(change_type, text_color)

                prefix_map = {
                    "added": self.tr("Added"),
                    "removed": self.tr("Removed"),
                    "modified": self.tr("Modified"),
                }
                prefix = prefix_map.get(change_type, change_type.capitalize())

                type_str = (
                    f", {self.tr('type')}={shape_type}" if shape_type else ""
                )
                if change_type == "modified":
                    extra = f" ({self.tr('points changed')})"
                else:
                    extra = ""
                html_parts.append(
                    f'<li style="color: {color};">'
                    f"{prefix}: {self.tr('label')}={label}{type_str}{extra}"
                    f"</li>"
                )
            html_parts.append("</ul>")
        elif img_diff.status != "unchanged":
            html_parts.append(
                f'<p style="color: {text_color}; font-style: italic;">'
                f'{self.tr("No shape-level details available.")}</p>'
            )

        self.detail_view.setHtml("\n".join(html_parts))


# ---------------------------------------------------------------------------
# VersionControlDialog (main dialog)
# ---------------------------------------------------------------------------


class VersionControlDialog(QDialog):
    """Main dialog for managing annotation version snapshots."""

    def __init__(self, version_manager, image_list, label_dir, parent=None):
        """
        Parameters
        ----------
        version_manager : VersionManager
            The version manager instance for the current project.
        image_list : list[str]
            List of image filenames in the project.
        label_dir : str
            Path to the directory containing annotation JSON files.
        parent : QWidget, optional
        """
        super().__init__(parent)
        self._vm = version_manager
        self._image_list = image_list
        self._label_dir = label_dir

        self.setWindowTitle(self.tr("Version Control"))
        self.setMinimumSize(820, 500)
        self.setModal(True)
        self.setWindowFlags(
            self.windowFlags()
            & ~Qt.WindowType.WindowContextHelpButtonHint
        )
        self.setStyleSheet(get_dialog_style())

        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(12)

        # ---- Top bar ----
        top_bar = QHBoxLayout()
        top_bar.setSpacing(8)

        self.create_btn = QPushButton(self.tr("Create Version"))
        self.create_btn.setStyleSheet(get_highlight_button_style())
        self.create_btn.clicked.connect(self._on_create_version)
        top_bar.addWidget(self.create_btn)

        self.refresh_btn = QPushButton(self.tr("Refresh"))
        self.refresh_btn.setStyleSheet(get_normal_button_style())
        self.refresh_btn.clicked.connect(self._refresh_table)
        top_bar.addWidget(self.refresh_btn)

        top_bar.addStretch()
        main_layout.addLayout(top_bar)

        # ---- Version table ----
        self.table = QTableWidget()
        self.table.setColumnCount(6)
        self.table.setHorizontalHeaderLabels([
            self.tr("Name"),
            self.tr("Timestamp"),
            self.tr("Images"),
            self.tr("Shapes"),
            self.tr("Description"),
            self.tr("Size"),
        ])
        self.table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows
        )
        self.table.setSelectionMode(
            QAbstractItemView.SelectionMode.ExtendedSelection
        )
        self.table.setEditTriggers(
            QAbstractItemView.EditTrigger.NoEditTriggers
        )
        self.table.verticalHeader().setVisible(False)
        self.table.setAlternatingRowColors(True)
        self.table.doubleClicked.connect(self._on_row_double_clicked)
        self.table.itemSelectionChanged.connect(self._update_button_states)

        # Column sizing
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(4, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(5, QHeaderView.ResizeMode.ResizeToContents)

        main_layout.addWidget(self.table, 1)

        # ---- Bottom button bar ----
        bottom_bar = QHBoxLayout()
        bottom_bar.setSpacing(8)

        self.compare_btn = QPushButton(self.tr("Compare"))
        self.compare_btn.setStyleSheet(get_ok_btn_style())
        self.compare_btn.setEnabled(False)
        self.compare_btn.setToolTip(
            self.tr("Select exactly 2 versions to compare")
        )
        self.compare_btn.clicked.connect(self._on_compare)
        bottom_bar.addWidget(self.compare_btn)

        self.restore_btn = QPushButton(self.tr("Restore"))
        self.restore_btn.setStyleSheet(get_normal_button_style())
        self.restore_btn.setEnabled(False)
        self.restore_btn.setToolTip(
            self.tr("Select exactly 1 version to restore")
        )
        self.restore_btn.clicked.connect(self._on_restore)
        bottom_bar.addWidget(self.restore_btn)

        self.delete_btn = QPushButton(self.tr("Delete"))
        self.delete_btn.setStyleSheet(_get_delete_btn_style())
        self.delete_btn.setEnabled(False)
        self.delete_btn.setToolTip(
            self.tr("Select one or more versions to delete")
        )
        self.delete_btn.clicked.connect(self._on_delete)
        bottom_bar.addWidget(self.delete_btn)

        bottom_bar.addStretch()

        self.close_btn = QPushButton(self.tr("Close"))
        self.close_btn.setStyleSheet(get_cancel_btn_style())
        self.close_btn.clicked.connect(self.accept)
        bottom_bar.addWidget(self.close_btn)

        main_layout.addLayout(bottom_bar)

        self.setLayout(main_layout)

        # --- Initial data load ---
        self._refresh_table()

    # -- table population --

    def _refresh_table(self):
        """Reload the version list from the version manager."""
        self.table.setRowCount(0)
        try:
            versions = self._vm.list_versions()
        except Exception as exc:
            logger.error("Failed to list versions: %s", exc)
            QMessageBox.critical(
                self,
                self.tr("Error"),
                self.tr("Failed to load versions: %s") % str(exc),
            )
            return

        self.table.setRowCount(len(versions))
        for row, info in enumerate(versions):
            # Name
            name_item = QTableWidgetItem(info.name)
            name_item.setData(Qt.ItemDataRole.UserRole, info)
            self.table.setItem(row, 0, name_item)

            # Timestamp
            ts_item = QTableWidgetItem(_format_timestamp(info.timestamp))
            ts_item.setTextAlignment(
                Qt.AlignmentFlag.AlignCenter
            )
            self.table.setItem(row, 1, ts_item)

            # Images
            stats = info.stats or {}
            images_text = str(stats.get("annotated_image_count", 0))
            total_images = stats.get("image_count", 0)
            if total_images:
                images_text = f"{images_text}/{total_images}"
            images_item = QTableWidgetItem(images_text)
            images_item.setTextAlignment(
                Qt.AlignmentFlag.AlignCenter
            )
            self.table.setItem(row, 2, images_item)

            # Shapes
            shapes_item = QTableWidgetItem(
                str(stats.get("total_shapes", 0))
            )
            shapes_item.setTextAlignment(
                Qt.AlignmentFlag.AlignCenter
            )
            self.table.setItem(row, 3, shapes_item)

            # Description
            desc_text = info.description if info.description else ""
            desc_item = QTableWidgetItem(desc_text)
            self.table.setItem(row, 4, desc_item)

            # Size
            size_item = QTableWidgetItem(_format_size(info.size_bytes))
            size_item.setTextAlignment(
                Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
            )
            self.table.setItem(row, 5, size_item)

        self._update_button_states()

    # -- button state management --

    def _update_button_states(self):
        """Enable or disable buttons based on the current selection."""
        selected_rows = self._get_selected_rows()
        count = len(selected_rows)
        self.compare_btn.setEnabled(count == 2)
        self.restore_btn.setEnabled(count == 1)
        self.delete_btn.setEnabled(count >= 1)

    def _get_selected_rows(self):
        """Return a list of unique selected row indices."""
        selection = self.table.selectionModel().selectedRows()
        return sorted(set(idx.row() for idx in selection))

    def _get_version_info(self, row):
        """Return the VersionInfo stored in the given table row."""
        item = self.table.item(row, 0)
        if item is None:
            return None
        return item.data(Qt.ItemDataRole.UserRole)

    # -- actions --

    def _on_create_version(self):
        """Open the create-version dialog and create a snapshot."""
        dlg = CreateVersionDialog(self)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return

        name = dlg.get_name()
        description = dlg.get_description()

        try:
            self._vm.create_version(
                name=name,
                description=description,
                image_list=self._image_list,
                label_dir=self._label_dir,
            )
            logger.info("Version '%s' created successfully.", name)
        except Exception as exc:
            logger.error("Failed to create version: %s", exc)
            QMessageBox.critical(
                self,
                self.tr("Error"),
                self.tr("Failed to create version: %s") % str(exc),
            )
            return

        self._refresh_table()

    def _on_compare(self):
        """Compare the two selected versions."""
        rows = self._get_selected_rows()
        if len(rows) != 2:
            return

        info_a = self._get_version_info(rows[0])
        info_b = self._get_version_info(rows[1])
        if info_a is None or info_b is None:
            return

        # The version listed first (newest) is displayed at row 0, so
        # comparing row0 vs row1 means "newer vs older".  We pass the
        # older version as the baseline (A) and newer as (B).
        try:
            diff = self._vm.compare_versions(
                info_b.version_id, info_a.version_id
            )
        except Exception as exc:
            logger.error("Failed to compare versions: %s", exc)
            QMessageBox.critical(
                self,
                self.tr("Error"),
                self.tr("Failed to compare versions: %s") % str(exc),
            )
            return

        dlg = VersionDiffDialog(diff, info_b, info_a, parent=self)
        dlg.exec()

    def _on_restore(self):
        """Restore annotations from the selected version."""
        rows = self._get_selected_rows()
        if len(rows) != 1:
            return

        info = self._get_version_info(rows[0])
        if info is None:
            return

        reply = QMessageBox.question(
            self,
            self.tr("Confirm Restore"),
            self.tr(
                "Are you sure you want to restore version '%s'?\n\n"
                "This will overwrite current annotation files in the "
                "label directory."
            )
            % info.name,
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        try:
            count = self._vm.restore_version(
                info.version_id, self._label_dir
            )
            logger.info(
                "Restored %d files from version '%s'.", count, info.name
            )
            QMessageBox.information(
                self,
                self.tr("Restore Complete"),
                self.tr("Successfully restored %n annotation file(s) from "
                        "version '%1'.", "", count)
                .replace("%1", info.name),
            )
        except Exception as exc:
            logger.error("Failed to restore version: %s", exc)
            QMessageBox.critical(
                self,
                self.tr("Error"),
                self.tr("Failed to restore version: %s") % str(exc),
            )

    def _on_delete(self):
        """Delete the selected version(s) after confirmation."""
        rows = self._get_selected_rows()
        if not rows:
            return

        infos = []
        for r in rows:
            info = self._get_version_info(r)
            if info is not None:
                infos.append(info)

        if not infos:
            return

        names = ", ".join(f"'{i.name}'" for i in infos)

        reply = QMessageBox.question(
            self,
            self.tr("Confirm Delete"),
            self.tr(
                "Are you sure you want to permanently delete the following "
                "version(s)?\n\n%s\n\nThis action cannot be undone."
            )
            % names,
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        errors = []
        for info in infos:
            try:
                ok = self._vm.delete_version(info.version_id)
                if ok:
                    logger.info("Deleted version '%s'.", info.name)
                else:
                    errors.append(info.name)
            except Exception as exc:
                logger.error(
                    "Failed to delete version '%s': %s", info.name, exc
                )
                errors.append(info.name)

        if errors:
            QMessageBox.warning(
                self,
                self.tr("Partial Failure"),
                self.tr("Could not delete the following version(s): %s")
                % ", ".join(errors),
            )

        self._refresh_table()

    def _on_row_double_clicked(self, index):
        """Show version details on double-click."""
        row = index.row()
        info = self._get_version_info(row)
        if info is None:
            return

        stats = info.stats or {}
        class_dist = stats.get("class_distribution", {})
        class_lines = []
        for cls, cnt in sorted(
            class_dist.items(), key=lambda x: x[1], reverse=True
        ):
            class_lines.append(f"  {cls}: {cnt}")

        class_text = "\n".join(class_lines) if class_lines else "  (none)"

        details = (
            f"{self.tr('Name')}: {info.name}\n"
            f"{self.tr('ID')}: {info.version_id}\n"
            f"{self.tr('Timestamp')}: {_format_timestamp(info.timestamp)}\n"
            f"{self.tr('Description')}: {info.description or '---'}\n\n"
            f"{self.tr('Total images')}: {stats.get('image_count', 0)}\n"
            f"{self.tr('Annotated images')}: "
            f"{stats.get('annotated_image_count', 0)}\n"
            f"{self.tr('Total shapes')}: {stats.get('total_shapes', 0)}\n"
            f"{self.tr('Size')}: {_format_size(info.size_bytes)}\n\n"
            f"{self.tr('Class distribution')}:\n{class_text}"
        )

        QMessageBox.information(
            self,
            self.tr("Version Details"),
            details,
        )
