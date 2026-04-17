"""Split management dialog for X-AnyLabeling.

Provides a comprehensive UI for managing train/val/test image splits,
including auto-split controls, per-split statistics, and manual
image assignment with multi-select support.
"""

import os.path as osp

from PyQt6 import QtCore, QtGui, QtWidgets
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QIcon, QPixmap, QPainter, QBrush
from PyQt6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QGroupBox,
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
    QDoubleSpinBox,
    QSpinBox,
    QComboBox,
    QPushButton,
    QLabel,
    QMenu,
    QLineEdit,
    QAbstractItemView,
    QListWidget,
    QListWidgetItem,
)

from anylabeling.views.labeling.logger import logger
from anylabeling.views.labeling.utils.split_indicators import (
    SPLIT_COLORS,
    get_split_icon,
)
from anylabeling.views.labeling.utils.style import (
    get_cancel_btn_style,
    get_dialog_style,
    get_ok_btn_style,
)

# Row-background colours for the statistics table (semi-transparent).
_STATS_ROW_COLORS = {
    "train": QColor(52, 152, 219, 35),       # light blue
    "val": QColor(243, 156, 18, 35),          # light orange
    "test": QColor(46, 204, 113, 35),         # light green
    "unassigned": QColor(149, 165, 166, 25),  # light gray
    "total": QColor(0, 0, 0, 0),              # transparent
}


class SplitManagerDialog(QDialog):
    """Dialog for managing train / val / test image splits.

    The dialog presents three sections:

    1. **Auto Split Controls** -- ratio spin-boxes, strategy selector,
       seed, and an *Auto Split* button.
    2. **Statistics Table** -- read-only overview of current split
       distribution including per-class counts.
    3. **Image Assignment List** -- searchable, multi-select list of
       images with right-click context menu for manual assignment.

    Args:
        split_manager: The project's :class:`SplitManager` instance.
        image_list: List of image file paths currently loaded.
        label_dir: Directory containing annotation JSON files.
        parent: Optional parent widget.
    """

    def __init__(self, split_manager, image_list, label_dir, parent=None):
        super().__init__(parent)
        self._split_manager = split_manager
        self._image_list = list(image_list)
        self._label_dir = label_dir

        # Keep basenames for split-manager lookups.
        self._basenames = [osp.basename(p) for p in self._image_list]

        # Snapshot so we can revert on Cancel.
        self._original_splits = split_manager.get_all_splits()

        self._init_ui()
        self._connect_signals()
        self._refresh_all()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _init_ui(self):
        """Build the complete dialog layout."""
        self.setWindowTitle(self.tr("Split Manager"))
        self.setMinimumSize(860, 640)
        self.resize(960, 720)
        self.setStyleSheet(get_dialog_style())

        root_layout = QVBoxLayout(self)
        root_layout.setSpacing(12)
        root_layout.setContentsMargins(16, 16, 16, 16)

        root_layout.addWidget(self._build_auto_split_group())
        root_layout.addWidget(self._build_stats_group(), 1)
        root_layout.addWidget(self._build_image_list_group(), 2)
        root_layout.addLayout(self._build_button_row())

    # -- auto-split controls --

    def _build_auto_split_group(self):
        group = QGroupBox(self.tr("Auto Split Controls"))
        layout = QHBoxLayout(group)
        layout.setSpacing(12)

        # --- ratio spin-boxes ---
        self._spin_train = self._make_ratio_spin(
            self.tr("Train:"), 0.70
        )
        self._spin_val = self._make_ratio_spin(
            self.tr("Val:"), 0.20
        )
        self._spin_test = self._make_ratio_spin(
            self.tr("Test:"), 0.10
        )

        for label_widget, spin in (
            self._spin_train,
            self._spin_val,
            self._spin_test,
        ):
            layout.addWidget(label_widget)
            layout.addWidget(spin)

        # Sum-warning label (hidden when sum == 1.0).
        self._lbl_sum_warning = QLabel()
        self._lbl_sum_warning.setStyleSheet("color: #e74c3c; font-size: 11px;")
        self._lbl_sum_warning.hide()
        layout.addWidget(self._lbl_sum_warning)

        layout.addSpacing(12)

        # --- strategy combo ---
        layout.addWidget(QLabel(self.tr("Strategy:")))
        self._combo_strategy = QComboBox()
        self._combo_strategy.addItem(self.tr("Random"), "random")
        self._combo_strategy.addItem(
            self.tr("Stratified (by class distribution)"), "stratified"
        )
        self._combo_strategy.setMinimumWidth(180)
        layout.addWidget(self._combo_strategy)

        layout.addSpacing(8)

        # --- seed ---
        layout.addWidget(QLabel(self.tr("Seed:")))
        self._spin_seed = QSpinBox()
        self._spin_seed.setRange(0, 999999)
        self._spin_seed.setValue(0)
        self._spin_seed.setToolTip(
            self.tr("0 = random seed, any other value for reproducibility")
        )
        self._spin_seed.setMinimumWidth(80)
        layout.addWidget(self._spin_seed)

        layout.addSpacing(12)

        # --- auto-split button ---
        self._btn_auto_split = QPushButton(self.tr("Auto Split"))
        self._btn_auto_split.setStyleSheet(get_ok_btn_style())
        self._btn_auto_split.setMinimumWidth(110)
        layout.addWidget(self._btn_auto_split)

        layout.addStretch()
        return group

    def _make_ratio_spin(self, label_text, default):
        """Create a (QLabel, QDoubleSpinBox) pair for a ratio value."""
        label = QLabel(label_text)
        spin = QDoubleSpinBox()
        spin.setRange(0.0, 1.0)
        spin.setSingleStep(0.05)
        spin.setDecimals(2)
        spin.setValue(default)
        spin.setMinimumWidth(72)
        return label, spin

    # -- statistics table --

    def _build_stats_group(self):
        group = QGroupBox(self.tr("Split Statistics"))
        layout = QVBoxLayout(group)

        self._stats_table = QTableWidget()
        self._stats_table.setEditTriggers(
            QAbstractItemView.EditTrigger.NoEditTriggers
        )
        self._stats_table.setSelectionMode(
            QAbstractItemView.SelectionMode.NoSelection
        )
        self._stats_table.verticalHeader().setVisible(False)
        self._stats_table.horizontalHeader().setStretchLastSection(True)
        self._stats_table.setAlternatingRowColors(False)

        layout.addWidget(self._stats_table)
        return group

    # -- image assignment list --

    def _build_image_list_group(self):
        group = QGroupBox(self.tr("Image Assignments"))
        layout = QVBoxLayout(group)

        # Search bar.
        search_layout = QHBoxLayout()
        search_layout.addWidget(QLabel(self.tr("Filter:")))
        self._search_edit = QLineEdit()
        self._search_edit.setPlaceholderText(
            self.tr("Type to filter images...")
        )
        self._search_edit.setClearButtonEnabled(True)
        search_layout.addWidget(self._search_edit)
        layout.addLayout(search_layout)

        # Image list widget with multi-select.
        self._image_list_widget = QListWidget()
        self._image_list_widget.setSelectionMode(
            QAbstractItemView.SelectionMode.ExtendedSelection
        )
        self._image_list_widget.setContextMenuPolicy(
            Qt.ContextMenuPolicy.CustomContextMenu
        )
        layout.addWidget(self._image_list_widget)

        return group

    # -- dialog buttons --

    def _build_button_row(self):
        layout = QHBoxLayout()
        layout.addStretch()

        self._btn_cancel = QPushButton(self.tr("Cancel"))
        self._btn_cancel.setStyleSheet(get_cancel_btn_style())
        layout.addWidget(self._btn_cancel)

        self._btn_apply = QPushButton(self.tr("Apply"))
        self._btn_apply.setStyleSheet(get_ok_btn_style())
        layout.addWidget(self._btn_apply)

        return layout

    # ------------------------------------------------------------------
    # Signal connections
    # ------------------------------------------------------------------

    def _connect_signals(self):
        # Ratio spin-boxes -> validate sum.
        self._spin_train[1].valueChanged.connect(self._on_ratio_changed)
        self._spin_val[1].valueChanged.connect(self._on_ratio_changed)
        self._spin_test[1].valueChanged.connect(self._on_ratio_changed)

        # Auto-split button.
        self._btn_auto_split.clicked.connect(self._on_auto_split)

        # Search bar.
        self._search_edit.textChanged.connect(self._on_search_changed)

        # Right-click context menu on image list.
        self._image_list_widget.customContextMenuRequested.connect(
            self._on_image_context_menu
        )

        # Dialog buttons.
        self._btn_apply.clicked.connect(self._on_apply)
        self._btn_cancel.clicked.connect(self._on_cancel)

    # ------------------------------------------------------------------
    # Slot implementations
    # ------------------------------------------------------------------

    def _on_ratio_changed(self):
        """Check whether the three ratios sum to 1.0 and show a warning
        if they do not."""
        total = (
            self._spin_train[1].value()
            + self._spin_val[1].value()
            + self._spin_test[1].value()
        )
        if abs(total - 1.0) > 1e-3:
            self._lbl_sum_warning.setText(
                self.tr("Sum = {:.2f} (must be 1.00)").format(total)
            )
            self._lbl_sum_warning.show()
            self._btn_auto_split.setEnabled(False)
        else:
            self._lbl_sum_warning.hide()
            self._btn_auto_split.setEnabled(True)

    def _on_auto_split(self):
        """Run the auto-split algorithm with the current UI parameters."""
        ratios = {
            "train": round(self._spin_train[1].value(), 2),
            "val": round(self._spin_val[1].value(), 2),
            "test": round(self._spin_test[1].value(), 2),
        }

        strategy = self._combo_strategy.currentData() or "random"
        seed_value = self._spin_seed.value()
        seed = seed_value if seed_value != 0 else None

        try:
            self._split_manager.auto_split(
                image_list=self._basenames,
                ratios=ratios,
                strategy=strategy,
                label_dir=self._label_dir,
                seed=seed,
            )
        except ValueError as e:
            logger.warning("Auto-split failed: %s", e)
            QtWidgets.QMessageBox.warning(
                self,
                self.tr("Auto Split Error"),
                str(e),
            )
            return

        self._refresh_all()

    def _on_search_changed(self, text):
        """Filter the image list to show only items matching *text*."""
        needle = text.lower()
        for idx in range(self._image_list_widget.count()):
            item = self._image_list_widget.item(idx)
            if item is None:
                continue
            item.setHidden(needle not in item.text().lower())

    def _on_image_context_menu(self, pos):
        """Show a context menu to assign selected images to a split."""
        selected = self._image_list_widget.selectedItems()
        if not selected:
            return

        menu = QMenu(self)

        action_train = menu.addAction(self.tr("Move to Train"))
        action_val = menu.addAction(self.tr("Move to Val"))
        action_test = menu.addAction(self.tr("Move to Test"))
        menu.addSeparator()
        action_unassigned = menu.addAction(self.tr("Move to Unassigned"))

        action = menu.exec(
            self._image_list_widget.viewport().mapToGlobal(pos)
        )
        if action is None:
            return

        target = None
        if action is action_train:
            target = "train"
        elif action is action_val:
            target = "val"
        elif action is action_test:
            target = "test"
        elif action is action_unassigned:
            target = "unassigned"

        if target is None:
            return

        filenames = []
        for item in selected:
            basename = item.data(Qt.ItemDataRole.UserRole)
            if basename:
                filenames.append(basename)

        if not filenames:
            return

        try:
            self._split_manager.set_partitions_batch(filenames, target)
        except ValueError as e:
            logger.warning("Batch assignment failed: %s", e)
            return

        self._refresh_all()

    def _on_apply(self):
        """Save splits to disk and close the dialog."""
        try:
            self._split_manager.save_splits()
        except Exception as e:
            logger.error("Failed to save splits: %s", e)
            QtWidgets.QMessageBox.warning(
                self,
                self.tr("Save Error"),
                self.tr("Could not save splits: {}").format(e),
            )
            return
        self.accept()

    def _on_cancel(self):
        """Restore the original split state and close the dialog."""
        try:
            # Revert to the snapshot taken at dialog open.
            for name in self._split_manager.SPLIT_NAMES:
                self._split_manager._splits[name] = list(
                    self._original_splits.get(name, [])
                )
        except Exception as e:
            logger.warning("Failed to revert splits on cancel: %s", e)
        self.reject()

    # ------------------------------------------------------------------
    # Refresh helpers
    # ------------------------------------------------------------------

    def _refresh_all(self):
        """Refresh both the statistics table and the image list."""
        self._refresh_stats_table()
        self._refresh_image_list()

    # -- statistics table --

    def _refresh_stats_table(self):
        """Recompute statistics and rebuild the stats table contents."""
        try:
            stats = self._split_manager.get_split_stats(self._label_dir)
        except Exception as e:
            logger.warning("Could not compute split stats: %s", e)
            return

        # Collect the union of all class names across splits.
        all_classes = set()
        for s in stats:
            all_classes.update(s.class_distribution.keys())
        all_classes = sorted(all_classes)

        # Fixed columns + one per class.
        fixed_cols = [
            self.tr("Split"),
            self.tr("Images"),
            self.tr("Shapes"),
            self.tr("Percentage"),
        ]
        headers = fixed_cols + all_classes
        num_cols = len(headers)

        # Rows: one per split + total.
        row_count = len(stats) + 1
        self._stats_table.setRowCount(row_count)
        self._stats_table.setColumnCount(num_cols)
        self._stats_table.setHorizontalHeaderLabels(headers)

        # Totals accumulators.
        total_images = 0
        total_shapes = 0
        total_class = {c: 0 for c in all_classes}

        for row, s in enumerate(stats):
            total_images += s.image_count
            total_shapes += s.shape_count
            bg = _STATS_ROW_COLORS.get(s.split_name, QColor(0, 0, 0, 0))

            self._set_stats_cell(row, 0, s.split_name, bg)
            self._set_stats_cell(row, 1, str(s.image_count), bg)
            self._set_stats_cell(row, 2, str(s.shape_count), bg)
            self._set_stats_cell(
                row, 3, "{:.1f}%".format(s.percentage), bg
            )

            for ci, cls_name in enumerate(all_classes):
                count = s.class_distribution.get(cls_name, 0)
                total_class[cls_name] += count
                self._set_stats_cell(
                    row, 4 + ci, str(count) if count else "", bg
                )

        # Total row.
        total_row = row_count - 1
        bg_total = _STATS_ROW_COLORS["total"]
        self._set_stats_cell(total_row, 0, self.tr("Total"), bg_total, bold=True)
        self._set_stats_cell(total_row, 1, str(total_images), bg_total, bold=True)
        self._set_stats_cell(total_row, 2, str(total_shapes), bg_total, bold=True)
        pct = 100.0 if total_images > 0 else 0.0
        self._set_stats_cell(
            total_row, 3, "{:.1f}%".format(pct), bg_total, bold=True
        )
        for ci, cls_name in enumerate(all_classes):
            self._set_stats_cell(
                total_row,
                4 + ci,
                str(total_class[cls_name]),
                bg_total,
                bold=True,
            )

        # Resize columns to contents, then let last stretch.
        self._stats_table.resizeColumnsToContents()
        header = self._stats_table.horizontalHeader()
        if header is not None:
            header.setStretchLastSection(True)

    def _set_stats_cell(self, row, col, text, bg_color, bold=False):
        """Create a read-only, centred table-widget item."""
        item = QTableWidgetItem(text)
        item.setTextAlignment(
            Qt.AlignmentFlag.AlignCenter | Qt.AlignmentFlag.AlignVCenter
        )
        item.setFlags(
            Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable
        )
        if bg_color and bg_color.alpha() > 0:
            item.setBackground(QBrush(bg_color))
        if bold:
            font = item.font()
            font.setBold(True)
            item.setFont(font)
        self._stats_table.setItem(row, col, item)

    # -- image assignment list --

    def _refresh_image_list(self):
        """Rebuild the image list widget with current split assignments."""
        # Remember scroll position.
        scrollbar = self._image_list_widget.verticalScrollBar()
        scroll_pos = scrollbar.value() if scrollbar else 0

        self._image_list_widget.clear()

        for basename in self._basenames:
            partition = self._split_manager.get_partition(basename)
            display = "{} [{}]".format(basename, partition)

            item = QListWidgetItem(get_split_icon(partition), display)
            item.setData(Qt.ItemDataRole.UserRole, basename)

            # Subtle foreground tint.
            fg = SPLIT_COLORS.get(partition, SPLIT_COLORS["unassigned"])
            item.setForeground(QBrush(fg))

            self._image_list_widget.addItem(item)

        # Restore scroll position.
        if scrollbar:
            scrollbar.setValue(scroll_pos)

        # Re-apply current search filter.
        current_filter = self._search_edit.text()
        if current_filter:
            self._on_search_changed(current_filter)
