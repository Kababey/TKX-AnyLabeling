"""Visual indicator utilities for split assignments in the file list widget.

Provides colored icons and foreground-color helpers so users can see
at a glance which train/val/test split every image belongs to.
"""

import os.path as osp

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QIcon, QPixmap, QPainter, QBrush

from anylabeling.views.labeling.logger import logger

# Partition name -> display colour mapping.
SPLIT_COLORS = {
    "train": QColor(52, 152, 219),      # Blue
    "val": QColor(243, 156, 18),         # Orange
    "test": QColor(46, 204, 113),        # Green
    "unassigned": QColor(149, 165, 166), # Gray
}

# Custom item-data role used to store the partition name on list items.
SPLIT_ROLE = Qt.ItemDataRole.UserRole + 1


def get_split_icon(partition: str) -> QIcon:
    """Create a small coloured circle icon for *partition*.

    Args:
        partition: One of ``"train"``, ``"val"``, ``"test"``,
                   or ``"unassigned"``.

    Returns:
        A 12x12 :class:`QIcon` filled with the partition colour.
    """
    color = SPLIT_COLORS.get(partition, SPLIT_COLORS["unassigned"])
    pixmap = QPixmap(12, 12)
    pixmap.fill(QColor(0, 0, 0, 0))  # transparent background

    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    painter.setBrush(QBrush(color))
    painter.setPen(Qt.PenStyle.NoPen)
    painter.drawEllipse(1, 1, 10, 10)
    painter.end()

    return QIcon(pixmap)


def update_file_list_split_indicators(file_list_widget, split_manager):
    """Apply visual indicators to every item in *file_list_widget*.

    For each item the function:

    1. Extracts the basename from ``item.text()``.
    2. Queries *split_manager* for the partition.
    3. Sets a coloured-circle icon and stores the partition name
       under :data:`SPLIT_ROLE` for programmatic access.

    Args:
        file_list_widget: A :class:`QListWidget` (or compatible) that
            contains one item per image file.
        split_manager: A :class:`SplitManager` instance that has been
            synchronised with the current image list.
    """
    if file_list_widget is None or split_manager is None:
        return

    try:
        for idx in range(file_list_widget.count()):
            item = file_list_widget.item(idx)
            if item is None:
                continue

            filename = osp.basename(item.text())
            partition = split_manager.get_partition(filename)

            item.setIcon(get_split_icon(partition))
            item.setData(SPLIT_ROLE, partition)
    except Exception as e:
        logger.warning("Failed to update split indicators: %s", e)


def clear_split_indicators(file_list_widget):
    """Remove all split indicators from *file_list_widget* items.

    Clears the icon and the stored partition data for every item.

    Args:
        file_list_widget: A :class:`QListWidget` (or compatible).
    """
    if file_list_widget is None:
        return

    try:
        for idx in range(file_list_widget.count()):
            item = file_list_widget.item(idx)
            if item is None:
                continue

            item.setIcon(QIcon())
            item.setData(SPLIT_ROLE, None)
    except Exception as e:
        logger.warning("Failed to clear split indicators: %s", e)
