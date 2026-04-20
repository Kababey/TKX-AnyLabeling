"""Dialog that maps incoming class names to existing project classes."""

from typing import Dict, List

from PyQt6 import QtWidgets
from PyQt6.QtWidgets import (
    QComboBox,
    QDialog,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from anylabeling.views.labeling.utils.style import (
    get_cancel_btn_style,
    get_ok_btn_style,
)


_ADD_AS_NEW_KEY = "__add_as_new__"


class ClassMappingDialog(QDialog):
    """Ask the user how to map each incoming class name.

    For every incoming class the user picks either an existing class to
    merge into, or ``<add as new>`` to keep the incoming name verbatim.
    """

    def __init__(
        self,
        incoming: List[str],
        existing: List[str],
        parent=None,
    ):
        super().__init__(parent)
        self.setWindowTitle(self.tr("Map incoming classes"))
        self.setMinimumWidth(520)

        self._incoming = list(incoming)
        self._existing = list(existing)
        self._combos: Dict[str, QComboBox] = {}

        outer = QVBoxLayout(self)
        outer.setContentsMargins(16, 16, 16, 16)
        outer.setSpacing(12)

        hint = QLabel(
            self.tr(
                "For each incoming class, choose an existing project "
                "class to merge into, or keep it as a new class."
            )
        )
        hint.setWordWrap(True)
        outer.addWidget(hint)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        inner = QWidget()
        grid = QGridLayout(inner)
        grid.setContentsMargins(4, 4, 4, 4)
        grid.setHorizontalSpacing(16)
        grid.setVerticalSpacing(8)

        grid.addWidget(QLabel(self.tr("Incoming")), 0, 0)
        grid.addWidget(QLabel(self.tr("Target")), 0, 1)

        add_as_new_label = self.tr("<add as new>")
        for row, name in enumerate(self._incoming, start=1):
            grid.addWidget(QLabel(name), row, 0)
            combo = QComboBox()
            for existing_name in self._existing:
                combo.addItem(existing_name, existing_name)
            combo.addItem(add_as_new_label, _ADD_AS_NEW_KEY)
            if name in self._existing:
                combo.setCurrentIndex(self._existing.index(name))
            else:
                combo.setCurrentIndex(combo.count() - 1)
            grid.addWidget(combo, row, 1)
            self._combos[name] = combo

        inner.setLayout(grid)
        scroll.setWidget(inner)
        outer.addWidget(scroll, 1)

        btn_row = QHBoxLayout()
        btn_row.addStretch(1)
        cancel_btn = QPushButton(self.tr("Cancel"))
        cancel_btn.setStyleSheet(get_cancel_btn_style())
        cancel_btn.clicked.connect(self.reject)
        ok_btn = QPushButton(self.tr("Apply mapping"))
        ok_btn.setStyleSheet(get_ok_btn_style())
        ok_btn.setDefault(True)
        ok_btn.clicked.connect(self.accept)
        btn_row.addWidget(cancel_btn)
        btn_row.addWidget(ok_btn)
        outer.addLayout(btn_row)

    def result_mapping(self) -> Dict[str, str]:
        """Return {incoming_name: final_name}.

        ``<add as new>`` is resolved to the original incoming name so
        callers never have to special-case the sentinel.
        """
        mapping: Dict[str, str] = {}
        for name, combo in self._combos.items():
            data = combo.currentData()
            if data == _ADD_AS_NEW_KEY or data is None:
                mapping[name] = name
            else:
                mapping[name] = str(data)
        return mapping
