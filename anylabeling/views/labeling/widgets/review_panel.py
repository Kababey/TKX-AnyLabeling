"""Bottom-right review/triage panel for managing labeled vs unlabeled data.

Sits above the Files list. Provides:
  - Counts (To-do / Labeled / Negative / Approved)
  - Status filter to hide/show items
  - Mark current as Negative sample (writes an empty annotation)
  - Approve selected / Approve all reviewed (copies image + label into
    the active project's images/ and annotations/ folders and records
    the basename in the project manifest)
"""

from PyQt6 import QtCore, QtWidgets
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from anylabeling.views.labeling.utils import review_manager as rm


FILTER_ITEMS = [
    ("All", "all"),
    ("To-do (unlabeled)", rm.TODO),
    ("Labeled", rm.LABELED),
    ("Negative", rm.NEGATIVE),
    ("Approved", rm.APPROVED),
    ("Not approved", "not_approved"),
]


class ReviewPanel(QWidget):
    """Compact toolbar+counts widget for triage of the file list.

    The panel holds no state of its own — it calls back into ``host``
    (the LabelingWidget) for every action and reads counts via
    ``host.review_counts()``.
    """

    def __init__(self, host, parent=None):
        super().__init__(parent)
        self._host = host

        outer = QVBoxLayout(self)
        outer.setContentsMargins(6, 4, 6, 4)
        outer.setSpacing(3)

        # ── Counts row ────────────────────────────────────────────
        self._counts_lbl = QLabel("")
        self._counts_lbl.setStyleSheet(
            "QLabel { font-size: 11px; color: #ccc; padding: 1px 2px; }"
        )
        self._counts_lbl.setWordWrap(True)
        outer.addWidget(self._counts_lbl)

        # ── Filter row ────────────────────────────────────────────
        filter_row = QHBoxLayout()
        filter_row.setSpacing(4)
        filter_row.addWidget(QLabel("Show:"))
        self._filter_combo = QComboBox()
        for label, _value in FILTER_ITEMS:
            self._filter_combo.addItem(label)
        self._filter_combo.currentIndexChanged.connect(self._on_filter_changed)
        filter_row.addWidget(self._filter_combo, 1)
        outer.addLayout(filter_row)

        # ── Action buttons ────────────────────────────────────────
        btn_row = QHBoxLayout()
        btn_row.setSpacing(4)
        self._neg_btn = QPushButton("Mark Negative")
        self._neg_btn.setToolTip(
            "Save an empty annotation for the selected image(s) so they "
            "count as reviewed negative samples."
        )
        self._neg_btn.clicked.connect(self._on_mark_negative)
        btn_row.addWidget(self._neg_btn)

        self._approve_sel_btn = QPushButton("Approve Selected")
        self._approve_sel_btn.setToolTip(
            "Copy selected reviewed images (labeled or negative) into the "
            "active project's dataset and mark them approved."
        )
        self._approve_sel_btn.clicked.connect(self._on_approve_selected)
        btn_row.addWidget(self._approve_sel_btn)

        self._approve_all_btn = QPushButton("Approve All Reviewed")
        self._approve_all_btn.setToolTip(
            "Approve every reviewed image (labeled or negative) that is "
            "not yet approved."
        )
        self._approve_all_btn.clicked.connect(self._on_approve_all)
        btn_row.addWidget(self._approve_all_btn)
        outer.addLayout(btn_row)

        self._status_lbl = QLabel("")
        self._status_lbl.setStyleSheet(
            "QLabel { font-size: 10px; color: #999; padding: 0 2px; }"
        )
        self._status_lbl.setWordWrap(True)
        outer.addWidget(self._status_lbl)

    # ── public API used by host ─────────────────────────────────

    def current_filter(self) -> str:
        idx = self._filter_combo.currentIndex()
        if 0 <= idx < len(FILTER_ITEMS):
            return FILTER_ITEMS[idx][1]
        return "all"

    def update_counts(self, counts: dict) -> None:
        self._counts_lbl.setText(
            f"To-do: <b>{counts.get(rm.TODO, 0)}</b>  "
            f"·  Labeled: <b>{counts.get(rm.LABELED, 0)}</b>  "
            f"·  Negative: <b>{counts.get(rm.NEGATIVE, 0)}</b>  "
            f"·  Approved: <b>{counts.get(rm.APPROVED, 0)}</b>"
        )

    def set_status(self, text: str) -> None:
        self._status_lbl.setText(text)

    # ── slots ───────────────────────────────────────────────────

    def _on_filter_changed(self, _idx: int) -> None:
        self._host.apply_review_filter(self.current_filter())

    def _on_mark_negative(self) -> None:
        self._host.mark_selection_negative()

    def _on_approve_selected(self) -> None:
        self._host.approve_selected_images()

    def _on_approve_all(self) -> None:
        self._host.approve_all_reviewed()
