"""Dataset health dashboard - shows statistics about the current dataset:
class distribution, annotation coverage, resolution stats, etc.
"""

import json
import os
import os.path as osp
from collections import Counter, defaultdict
from typing import Dict, List, Optional

from PyQt6 import QtCore
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QPainter, QBrush, QPen
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QGroupBox,
    QHeaderView,
    QWidget,
    QGridLayout,
    QScrollArea,
)

from anylabeling.views.labeling.logger import logger
from anylabeling.views.labeling.utils.image_resizer import get_image_size
from anylabeling.views.labeling.utils.style import (
    get_cancel_btn_style,
    get_dialog_style,
)


class _BarChartWidget(QWidget):
    """Simple horizontal bar chart widget for class distribution."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._data: List[tuple] = []  # list of (label, count, color)
        self._max_count = 0
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMinimumHeight(200)

    def set_data(self, data: List[tuple]) -> None:
        self._data = data
        self._max_count = max((d[1] for d in data), default=0)
        self.update()

    def paintEvent(self, event):
        if not self._data:
            return
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        w = self.width()
        h = self.height()
        n = len(self._data)
        row_h = max(18, (h - 10) // max(n, 1))
        max_bar_w = w - 220
        label_area = 150
        value_area = 60
        top = 5

        for i, (label, count, color) in enumerate(self._data):
            y = top + i * row_h
            bar_h = row_h - 4
            bar_w = int(max_bar_w * (count / self._max_count)) if self._max_count else 0

            # Label text
            p.setPen(QColor("#333"))
            p.drawText(
                0, y, label_area, bar_h,
                Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignRight,
                label[:22] + ("..." if len(label) > 22 else "")
            )

            # Bar
            try:
                bar_color = QColor(color)
            except Exception:
                bar_color = QColor("#3498db")
            p.setBrush(QBrush(bar_color))
            p.setPen(Qt.PenStyle.NoPen)
            p.drawRoundedRect(
                label_area + 10, y + 2, bar_w, bar_h - 4, 3, 3
            )

            # Value
            p.setPen(QColor("#333"))
            p.drawText(
                label_area + 10 + bar_w + 5, y,
                value_area, bar_h,
                Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
                str(count),
            )


class DatasetHealthDialog(QDialog):
    """Dialog showing dataset statistics."""

    def __init__(
        self,
        image_list: List[str],
        annotation_dir: str,
        classes: Optional[List[Dict]] = None,
        parent=None,
    ):
        super().__init__(parent)
        self._image_list = image_list
        self._annotation_dir = annotation_dir
        self._classes = classes or []

        self.setWindowTitle(self.tr("Dataset Health"))
        self.setMinimumSize(800, 600)
        self.setStyleSheet(get_dialog_style())

        self._build_ui()
        self._compute_and_display()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(12)

        header = QLabel(self.tr("Dataset Health Dashboard"))
        header.setStyleSheet("font-size: 18px; font-weight: 600;")
        layout.addWidget(header)

        # Scroll area for the whole content
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        inner = QWidget()
        self._inner_layout = QVBoxLayout(inner)
        self._inner_layout.setSpacing(14)
        scroll.setWidget(inner)
        layout.addWidget(scroll, 1)

        # Close button
        btn_row = QHBoxLayout()
        btn_row.addStretch()
        refresh_btn = QPushButton(self.tr("Refresh"))
        refresh_btn.setStyleSheet(get_cancel_btn_style())
        refresh_btn.clicked.connect(self._compute_and_display)
        btn_row.addWidget(refresh_btn)
        close_btn = QPushButton(self.tr("Close"))
        close_btn.setStyleSheet(get_cancel_btn_style())
        close_btn.clicked.connect(self.accept)
        btn_row.addWidget(close_btn)
        layout.addLayout(btn_row)

    def _compute_and_display(self) -> None:
        # Clear existing
        while self._inner_layout.count():
            item = self._inner_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        stats = self._compute_stats()

        # === Summary cards ===
        summary_group = QGroupBox(self.tr("Summary"))
        summary_layout = QGridLayout(summary_group)
        cards = [
            (self.tr("Total images"), stats["total_images"]),
            (self.tr("Annotated"), f"{stats['annotated']} ({stats['annotated_pct']:.1f}%)"),
            (self.tr("Unannotated"), stats["unannotated"]),
            (self.tr("Total shapes"), stats["total_shapes"]),
            (self.tr("Avg shapes / image"), f"{stats['avg_shapes']:.2f}"),
            (self.tr("Class count"), stats["class_count"]),
        ]
        for i, (label, value) in enumerate(cards):
            card = QLabel(f"<b style='font-size:20px'>{value}</b><br/>"
                          f"<span style='color:#666'>{label}</span>")
            card.setTextFormat(Qt.TextFormat.RichText)
            card.setAlignment(Qt.AlignmentFlag.AlignCenter)
            card.setStyleSheet(
                "QLabel { background: #f5f5f5; border-radius: 8px;"
                " padding: 12px; }"
            )
            summary_layout.addWidget(card, i // 3, i % 3)
        self._inner_layout.addWidget(summary_group)

        # === Class distribution ===
        class_group = QGroupBox(self.tr("Class distribution"))
        class_layout = QVBoxLayout(class_group)
        if stats["class_distribution"]:
            # Chart
            chart = _BarChartWidget()
            chart_data = []
            class_colors = {
                c["name"]: c.get("color", "#3498db") for c in self._classes
            }
            for label, count in stats["class_distribution"].most_common():
                chart_data.append((
                    label, count, class_colors.get(label, "#3498db")
                ))
            chart.set_data(chart_data)
            chart.setMinimumHeight(min(500, 30 * len(chart_data) + 20))
            class_layout.addWidget(chart)

            # Table with detail
            table = QTableWidget(len(chart_data), 4)
            table.setHorizontalHeaderLabels([
                self.tr("Class"),
                self.tr("Instances"),
                self.tr("Images"),
                self.tr("% of shapes"),
            ])
            table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
            table.horizontalHeader().setSectionResizeMode(
                0, QHeaderView.ResizeMode.Stretch
            )
            for i, (label, count, _color) in enumerate(chart_data):
                table.setItem(i, 0, QTableWidgetItem(label))
                table.setItem(i, 1, QTableWidgetItem(str(count)))
                table.setItem(
                    i, 2,
                    QTableWidgetItem(str(stats["class_image_count"].get(label, 0))),
                )
                pct = count / stats["total_shapes"] * 100 if stats["total_shapes"] else 0
                table.setItem(i, 3, QTableWidgetItem(f"{pct:.1f}%"))
            table.setMaximumHeight(240)
            class_layout.addWidget(table)

            # Imbalance warning
            if len(chart_data) >= 2:
                max_c = chart_data[0][1]
                min_c = chart_data[-1][1]
                if min_c > 0 and max_c / min_c > 5:
                    warn = QLabel(self.tr(
                        "⚠ Class imbalance detected: most common class has "
                        "%.1fx more instances than the rarest."
                    ) % (max_c / min_c))
                    warn.setStyleSheet(
                        "color: #ed6c02; padding: 8px; background: #fff4e5;"
                        " border-radius: 4px;"
                    )
                    warn.setWordWrap(True)
                    class_layout.addWidget(warn)
        else:
            class_layout.addWidget(QLabel(self.tr("No annotations yet.")))
        self._inner_layout.addWidget(class_group)

        # === Resolution distribution ===
        res_group = QGroupBox(self.tr("Resolution distribution"))
        res_layout = QVBoxLayout(res_group)
        res_dist = stats["resolution_distribution"]
        if res_dist:
            table = QTableWidget(len(res_dist), 2)
            table.setHorizontalHeaderLabels([
                self.tr("Resolution"),
                self.tr("Count"),
            ])
            table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
            table.horizontalHeader().setSectionResizeMode(
                0, QHeaderView.ResizeMode.Stretch
            )
            for i, (size, count) in enumerate(res_dist.most_common()):
                size_str = f"{size[0]}x{size[1]}" if size else self.tr("Unreadable")
                table.setItem(i, 0, QTableWidgetItem(size_str))
                table.setItem(i, 1, QTableWidgetItem(str(count)))
            table.setMaximumHeight(180)
            res_layout.addWidget(table)

            if len(res_dist) > 1:
                warn = QLabel(self.tr(
                    "ⓘ Multiple resolutions found. Consider normalizing "
                    "image sizes for consistent training."
                ))
                warn.setStyleSheet(
                    "color: #1976d2; padding: 8px; background: #e3f2fd;"
                    " border-radius: 4px;"
                )
                warn.setWordWrap(True)
                res_layout.addWidget(warn)
        else:
            res_layout.addWidget(QLabel(self.tr("No images in dataset.")))
        self._inner_layout.addWidget(res_group)

        self._inner_layout.addStretch()

    def _compute_stats(self) -> Dict:
        total_images = len(self._image_list)
        annotated = 0
        total_shapes = 0
        class_dist = Counter()
        class_image_count = defaultdict(set)
        res_dist = Counter()

        for img_path in self._image_list:
            base = osp.splitext(osp.basename(img_path))[0]
            ann_path = osp.join(self._annotation_dir, base + ".json")
            # Also try alongside image
            if not osp.isfile(ann_path):
                alt = osp.join(osp.dirname(img_path), base + ".json")
                if osp.isfile(alt):
                    ann_path = alt

            shape_count = 0
            seen_classes_this_img = set()
            if osp.isfile(ann_path):
                try:
                    with open(ann_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    for s in data.get("shapes", []):
                        label = s.get("label")
                        if label and not str(label).startswith("AUTOLABEL_"):
                            class_dist[label] += 1
                            seen_classes_this_img.add(label)
                            shape_count += 1
                except (OSError, json.JSONDecodeError):
                    pass

            if shape_count > 0:
                annotated += 1
                total_shapes += shape_count
                for c in seen_classes_this_img:
                    class_image_count[c].add(img_path)

            # Resolution
            size = get_image_size(img_path)
            res_dist[size] += 1

        unannotated = total_images - annotated
        annotated_pct = (annotated / total_images * 100) if total_images else 0
        avg_shapes = (total_shapes / annotated) if annotated else 0

        return {
            "total_images": total_images,
            "annotated": annotated,
            "unannotated": unannotated,
            "annotated_pct": annotated_pct,
            "total_shapes": total_shapes,
            "avg_shapes": avg_shapes,
            "class_count": len(class_dist),
            "class_distribution": class_dist,
            "class_image_count": {k: len(v) for k, v in class_image_count.items()},
            "resolution_distribution": res_dist,
        }
