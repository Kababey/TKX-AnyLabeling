"""Data augmentation dialog – Random Crop, Window Filter, and Mixed Augmentation."""

import copy
import os
from pathlib import Path

import cv2
import numpy as np
from PyQt6 import QtCore, QtGui, QtWidgets
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
    QHeaderView,
    QSizePolicy,
    QFrame,
)

from anylabeling.views.labeling.utils.augmentation_engine import (
    DEFAULT_PRESETS,
    M1_PRESET,
    M2_PRESET,
    M3_PRESET,
    apply_filter,
    imread_unicode,
    load_image_filter_config,
    save_image_filter_config,
    delete_image_filter_config,
    run_filter_dataset,
    run_mixed_filter_augmentation,
    run_random_crop,
)
from anylabeling.views.labeling.utils.style import (
    get_cancel_btn_style,
    get_dialog_style,
    get_ok_btn_style,
)


# ── Worker threads ─────────────────────────────────────────────────────

class _CropWorker(QThread):
    progress = pyqtSignal(int)
    finished = pyqtSignal(dict)

    def __init__(self, params: dict):
        super().__init__()
        self._params = params

    def run(self):
        try:
            result = run_random_crop(
                **self._params, progress_callback=self.progress.emit
            )
        except Exception as exc:  # never crash the GUI thread
            result = {"error": f"{type(exc).__name__}: {exc}"}
        self.finished.emit(result)


class _FilterWorker(QThread):
    progress = pyqtSignal(int)
    finished = pyqtSignal(dict)

    def __init__(self, params: dict):
        super().__init__()
        self._params = params

    def run(self):
        result = run_filter_dataset(**self._params, progress_callback=self.progress.emit)
        self.finished.emit(result)


class _MixedAugWorker(QThread):
    progress = pyqtSignal(int)
    finished = pyqtSignal(dict)

    def __init__(self, params: dict):
        super().__init__()
        self._params = params

    def run(self):
        result = run_mixed_filter_augmentation(**self._params, progress_callback=self.progress.emit)
        self.finished.emit(result)


# ── Collapsible section widget ─────────────────────────────────────────

class CollapsibleSection(QWidget):
    """A titled, collapsible section widget."""

    def __init__(self, title: str, collapsed: bool = False, parent=None):
        super().__init__(parent)
        self._title = title

        self._toggle_btn = QPushButton()
        self._toggle_btn.setCheckable(True)
        self._toggle_btn.setChecked(not collapsed)
        self._toggle_btn.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._toggle_btn.setStyleSheet(
            "QPushButton { text-align:left; font-weight:bold; padding:5px 8px; "
            "border:none; border-radius:3px; background:#2d2d2d; color:#ddd; } "
            "QPushButton:hover { background:#3d3d3d; }"
        )
        self._update_btn_text()

        self._content = QWidget()
        self._content_layout = QVBoxLayout(self._content)
        self._content_layout.setContentsMargins(8, 4, 4, 6)
        self._content_layout.setSpacing(6)
        self._content.setVisible(not collapsed)

        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet("color: #444;")

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 4, 0, 0)
        outer.setSpacing(0)
        outer.addWidget(self._toggle_btn)
        outer.addWidget(self._content)
        outer.addWidget(sep)

        self._toggle_btn.toggled.connect(self._on_toggle)

    def _update_btn_text(self):
        arrow = "▼" if self._toggle_btn.isChecked() else "▶"
        self._toggle_btn.setText(f"{arrow}  {self._title}")

    def _on_toggle(self, checked: bool):
        self._content.setVisible(checked)
        self._update_btn_text()

    def add_widget(self, widget: QWidget):
        self._content_layout.addWidget(widget)

    def add_layout(self, layout):
        self._content_layout.addLayout(layout)

    def content_layout(self):
        return self._content_layout


# ── Helpers ────────────────────────────────────────────────────────────

def _labeled_slider(label_text: str, lo: int, hi: int, default: int, step: int = 1):
    row = QWidget()
    lay = QHBoxLayout(row)
    lay.setContentsMargins(0, 0, 0, 0)
    lbl = QLabel(label_text)
    lbl.setFixedWidth(175)
    slider = QSlider(Qt.Orientation.Horizontal)
    slider.setRange(lo, hi)
    slider.setValue(default)
    slider.setSingleStep(step)
    val_lbl = QLabel(str(default))
    val_lbl.setFixedWidth(42)
    val_lbl.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
    slider.valueChanged.connect(lambda v: val_lbl.setText(str(v)))
    lay.addWidget(lbl)
    lay.addWidget(slider)
    lay.addWidget(val_lbl)
    return row, slider, val_lbl


def _labeled_dspinbox(label_text: str, lo: float, hi: float, default: float, step: float = 0.01):
    row = QWidget()
    lay = QHBoxLayout(row)
    lay.setContentsMargins(0, 0, 0, 0)
    lbl = QLabel(label_text)
    lbl.setFixedWidth(175)
    spin = QDoubleSpinBox()
    spin.setRange(lo, hi)
    spin.setSingleStep(step)
    spin.setValue(default)
    spin.setDecimals(2)
    lay.addWidget(lbl)
    lay.addWidget(spin)
    return row, spin


def _int_row(label_text: str, lo: int, hi: int, default: int, label_width: int = 175):
    row = QWidget()
    lay = QHBoxLayout(row)
    lay.setContentsMargins(0, 0, 0, 0)
    lbl = QLabel(label_text)
    lbl.setFixedWidth(label_width)
    spin = QSpinBox()
    spin.setRange(lo, hi)
    spin.setValue(default)
    lay.addWidget(lbl)
    lay.addWidget(spin)
    return row, spin


def _dir_row(label_text: str, placeholder: str = "", label_width: int = 100):
    row = QWidget()
    lay = QHBoxLayout(row)
    lay.setContentsMargins(0, 0, 0, 0)
    lbl = QLabel(label_text)
    lbl.setFixedWidth(label_width)
    edit = QLineEdit()
    if placeholder:
        edit.setPlaceholderText(placeholder)
    btn = QPushButton("Browse…")
    btn.setFixedWidth(80)
    btn.setStyleSheet(get_cancel_btn_style())
    lay.addWidget(lbl)
    lay.addWidget(edit)
    lay.addWidget(btn)
    return row, edit, btn


def _browse_dir(parent, edit: QLineEdit):
    d = QFileDialog.getExistingDirectory(parent, "Select Directory", edit.text() or "")
    if d:
        edit.setText(d)


# ── Preview panel ──────────────────────────────────────────────────────

class _PreviewPanel(QWidget):
    """Displays up to 4 image tiles: Original | M1 | M2 | M3."""

    TITLES = ["Original", "M1", "M2", "M3"]
    TILE_SIZE = 200

    def __init__(self, parent=None):
        super().__init__(parent)
        grid = QtWidgets.QGridLayout(self)
        grid.setSpacing(6)
        self._labels = []
        for idx, title in enumerate(self.TITLES):
            col = idx % 2
            row_base = (idx // 2) * 2
            header = QLabel(title)
            header.setAlignment(Qt.AlignmentFlag.AlignCenter)
            header.setStyleSheet("font-weight: bold; font-size: 11px;")
            img_lbl = QLabel()
            img_lbl.setFixedSize(self.TILE_SIZE, self.TILE_SIZE)
            img_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            img_lbl.setStyleSheet("background: #1e1e1e; border: 1px solid #555;")
            grid.addWidget(header, row_base, col)
            grid.addWidget(img_lbl, row_base + 1, col)
            self._labels.append(img_lbl)

    def update_images(self, bgr_original: np.ndarray, settings: list, use_clahe: bool):
        images = [bgr_original]
        for c, w in settings:
            images.append(apply_filter(bgr_original, c, w, use_clahe))
        for label, img in zip(self._labels, images):
            self._set_pixmap(label, img)

    def clear(self):
        for lbl in self._labels:
            lbl.clear()
            lbl.setText("—")

    @staticmethod
    def _set_pixmap(label: QLabel, bgr: np.ndarray):
        if bgr is None:
            label.setText("—")
            return
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        max_side = _PreviewPanel.TILE_SIZE
        scale = min(max_side / w, max_side / h, 1.0)
        if scale < 1.0:
            new_w, new_h = int(w * scale), int(h * scale)
            rgb = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
        qimg = QtGui.QImage(
            rgb.data, rgb.shape[1], rgb.shape[0],
            rgb.shape[1] * 3, QtGui.QImage.Format.Format_RGB888,
        )
        label.setPixmap(QtGui.QPixmap.fromImage(qimg))


# ── Tab 1: Random Crop ─────────────────────────────────────────────────

class _RandomCropTab(QWidget):
    def __init__(self, default_dataset_dir: str = "", parent=None):
        super().__init__(parent)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)

        inner = QWidget()
        outer = QVBoxLayout(inner)
        outer.setContentsMargins(10, 10, 10, 10)
        outer.setSpacing(4)

        # Parameters section
        params_sec = CollapsibleSection("Augmentation Parameters")
        pg = params_sec.content_layout()
        pg.setSpacing(6)

        self._pct_row, self.pct_spin = _int_row(
            "Data to crop (% of labelled):", 1, 100, 30
        )
        self.pct_spin.setToolTip(
            "Randomly pick this percentage of LABELLED images for cropping.\n"
            "Empty images (negative samples / no labels) are never cropped,\n"
            "but are still copied as originals if that option is enabled."
        )
        pg.addWidget(self._pct_row)

        self._n_row, self.n_spin = _int_row("Augmentations per image:", 1, 50, 1)
        pg.addWidget(self._n_row)

        self._minr_row, self.min_ratio_spin = _labeled_dspinbox("Min crop ratio:", 0.05, 0.95, 0.30)
        pg.addWidget(self._minr_row)

        self._maxr_row, self.max_ratio_spin = _labeled_dspinbox("Max crop ratio:", 0.05, 0.95, 0.70)
        pg.addWidget(self._maxr_row)

        self._mmask_row, self.min_mask_spin = _labeled_dspinbox("Min mask ratio:", 0.01, 0.99, 0.10)
        pg.addWidget(self._mmask_row)

        quality_row, self.quality_spin = _int_row("JPEG quality (0-100):", 1, 100, 100)
        pg.addWidget(quality_row)

        seed_row, self.seed_spin = _int_row("Random seed:", 0, 999999, 42)
        pg.addWidget(seed_row)

        self.copy_orig_cb = QCheckBox("Copy originals to output")
        self.copy_orig_cb.setChecked(True)
        pg.addWidget(self.copy_orig_cb)

        outer.addWidget(params_sec)

        # I/O section
        io_sec = CollapsibleSection("Input / Output Directories")
        ig = io_sec.content_layout()

        self._ds_row, self.ds_edit, ds_btn = _dir_row("Dataset dir:", "Path to folder with images/ and labels/")
        if default_dataset_dir:
            self.ds_edit.setText(default_dataset_dir)
        ds_btn.clicked.connect(lambda: _browse_dir(self, self.ds_edit))
        ig.addWidget(self._ds_row)

        self._out_row, self.out_edit, out_btn = _dir_row("Output dir:", "Destination folder (will be created)")
        out_btn.clicked.connect(lambda: _browse_dir(self, self.out_edit))
        ig.addWidget(self._out_row)

        outer.addWidget(io_sec)

        # Progress + status
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.hide()
        outer.addWidget(self.progress_bar)

        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)
        outer.addWidget(self.status_label)

        btn_row = QHBoxLayout()
        btn_row.addStretch()
        self.run_btn = QPushButton("Run Random Crop Augmentation")
        self.run_btn.setStyleSheet(get_ok_btn_style())
        btn_row.addWidget(self.run_btn)
        outer.addLayout(btn_row)

        outer.addStretch()

        scroll.setWidget(inner)
        top = QVBoxLayout(self)
        top.setContentsMargins(0, 0, 0, 0)
        top.addWidget(scroll)

        self.run_btn.clicked.connect(self._run)
        self._worker = None

    def _run(self):
        ds = self.ds_edit.text().strip()
        out = self.out_edit.text().strip()
        if not ds or not out:
            QMessageBox.warning(self, "Missing Path", "Please set both dataset and output directories.")
            return
        if not Path(ds, "images").exists():
            QMessageBox.warning(self, "Invalid Dataset", f"No 'images' subfolder found in:\n{ds}")
            return

        params = {
            "dataset_dir": ds,
            "output_dir": out,
            "n_aug_per_image": self.n_spin.value(),
            "crop_min_ratio": self.min_ratio_spin.value(),
            "crop_max_ratio": self.max_ratio_spin.value(),
            "min_mask_ratio": self.min_mask_spin.value(),
            "copy_originals": self.copy_orig_cb.isChecked(),
            "output_jpeg_quality": self.quality_spin.value(),
            "data_fraction": self.pct_spin.value() / 100.0,
            "seed": self.seed_spin.value(),
        }

        self.run_btn.setEnabled(False)
        self.progress_bar.setValue(0)
        self.progress_bar.show()
        self.status_label.setText("Running…")

        self._worker = _CropWorker(params)
        self._worker.progress.connect(self.progress_bar.setValue)
        self._worker.finished.connect(self._on_done)
        self._worker.start()

    def _on_done(self, result: dict):
        self.run_btn.setEnabled(True)
        self.progress_bar.hide()
        if "error" in result:
            self.status_label.setText(f"Error: {result['error']}")
            QMessageBox.critical(self, "Augmentation Error", result["error"])
        else:
            backend = result.get("backend", "opencv")
            sel_line = ""
            if "eligible" in result:
                sel_line = (
                    f"Labelled images: {result.get('eligible', 0)} | "
                    f"Selected for crop: {result.get('selected', 0)}\n"
                )
            note = result.get("note")
            msg = (
                f"Done! Originals: {result['originals']} | "
                f"Augmented: {result['augmented']} | "
                f"Total: {result['total']}\n"
                f"{sel_line}"
                f"Output: {result['output_dir']}\n"
                f"Backend: {backend}"
            )
            if note:
                msg += f"\nNote: {note}"
            self.status_label.setText(msg)
            QMessageBox.information(self, "Complete", msg)


# ── Tab 2: Window Filter ───────────────────────────────────────────────

class _FilterTab(QWidget):
    def __init__(self, current_image_path: str = "", default_dataset_dir: str = "", parent=None):
        super().__init__(parent)
        self._current_image_path = current_image_path
        self._bgr_image = None
        self._worker = None
        self.setAcceptDrops(True)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)

        inner = QWidget()
        outer = QVBoxLayout(inner)
        outer.setContentsMargins(10, 10, 10, 10)
        outer.setSpacing(4)

        # ── Section 1: Preview ────────────────────────────────────────
        preview_sec = CollapsibleSection("Preview (current image)")
        pv = preview_sec.content_layout()

        self._preview = _PreviewPanel()
        pv.addWidget(self._preview)

        drop_hint = QLabel(
            "Tip: drag an image from the Files list (bottom-right of the "
            "main window) and drop it here to preview it."
        )
        drop_hint.setWordWrap(True)
        drop_hint.setStyleSheet("color:#888; font-style:italic;")
        pv.addWidget(drop_hint)

        load_row = QHBoxLayout()
        load_row.addWidget(QLabel("Image:"))
        self._img_path_edit = QLineEdit(current_image_path)
        self._img_path_edit.setPlaceholderText("Image path for preview…")
        load_row.addWidget(self._img_path_edit)
        load_btn = QPushButton("Load")
        load_btn.setFixedWidth(55)
        load_btn.setStyleSheet(get_cancel_btn_style())
        load_btn.clicked.connect(self._load_preview_image)
        load_row.addWidget(load_btn)
        browse_img_btn = QPushButton("Browse…")
        browse_img_btn.setFixedWidth(75)
        browse_img_btn.setStyleSheet(get_cancel_btn_style())
        browse_img_btn.clicked.connect(self._browse_image)
        load_row.addWidget(browse_img_btn)
        pv.addLayout(load_row)

        outer.addWidget(preview_sec)

        # ── Section 2: Filter Settings ────────────────────────────────
        settings_sec = CollapsibleSection("Filter Settings")
        sg = settings_sec.content_layout()
        sg.setSpacing(6)

        self._clahe_cb = QCheckBox("Apply CLAHE on top of window filter")
        self._clahe_cb.toggled.connect(self._refresh_preview)
        sg.addWidget(self._clahe_cb)

        preset_row = QHBoxLayout()
        preset_row.addWidget(QLabel("Quick presets:"))
        for name, (c, w) in [("M1", M1_PRESET), ("M2", M2_PRESET), ("M3", M3_PRESET)]:
            btn = QPushButton(f"{name}  (C={c}, W={w})")
            btn.setStyleSheet(get_cancel_btn_style())
            btn.clicked.connect(lambda checked, _c=c, _w=w: self._apply_preset(_c, _w))
            preset_row.addWidget(btn)
        preset_row.addStretch()
        sg.addLayout(preset_row)

        # Preset editor (M1 / M2 / M3 spinboxes)
        preset_edit_grp = QGroupBox("Customize Presets (used in preview & dataset ops)")
        peg = QtWidgets.QGridLayout(preset_edit_grp)
        peg.setSpacing(6)
        preset_data = [("M1", M1_PRESET), ("M2", M2_PRESET), ("M3", M3_PRESET)]
        self._preset_c_spins = {}
        self._preset_w_spins = {}
        for col, (name, (c, w)) in enumerate(preset_data):
            peg.addWidget(QLabel(name), 0, col * 3, Qt.AlignmentFlag.AlignCenter)
            peg.addWidget(QLabel("C:"), 1, col * 3)
            c_spin = QSpinBox()
            c_spin.setRange(0, 255)
            c_spin.setValue(c)
            c_spin.valueChanged.connect(self._refresh_preview)
            peg.addWidget(c_spin, 1, col * 3 + 1)
            peg.addWidget(QLabel("W:"), 2, col * 3)
            w_spin = QSpinBox()
            w_spin.setRange(1, 510)
            w_spin.setValue(w)
            w_spin.valueChanged.connect(self._refresh_preview)
            peg.addWidget(w_spin, 2, col * 3 + 1)
            self._preset_c_spins[name] = c_spin
            self._preset_w_spins[name] = w_spin
        sg.addWidget(preset_edit_grp)
        outer.addWidget(settings_sec)

        # ── Section 3: Per-Image Config ───────────────────────────────
        img_cfg_sec = CollapsibleSection("Per-Image Filter Config", collapsed=True)
        ic = img_cfg_sec.content_layout()
        ic.setSpacing(6)

        ic.addWidget(QLabel(
            "Save/load filter settings specifically for the current previewed image.\n"
            "These configs can be used to track intended windowing per image."
        ))

        # Active single-image filter controls
        single_row_c, self._single_c_spin = _int_row("Center:", 0, 255, 128, 80)
        single_row_w, self._single_w_spin = _int_row("Width:", 1, 510, 110, 80)
        self._single_clahe_cb = QCheckBox("CLAHE")
        ic.addWidget(single_row_c)
        ic.addWidget(single_row_w)
        ic.addWidget(self._single_clahe_cb)

        ic_btn_row = QHBoxLayout()
        save_cfg_btn = QPushButton("Save Config for This Image")
        save_cfg_btn.setStyleSheet(get_ok_btn_style())
        save_cfg_btn.clicked.connect(self._save_img_filter_config)
        load_cfg_btn = QPushButton("Load Saved Config")
        load_cfg_btn.setStyleSheet(get_cancel_btn_style())
        load_cfg_btn.clicked.connect(self._load_img_filter_config)
        del_cfg_btn = QPushButton("Delete Config")
        del_cfg_btn.setStyleSheet(get_cancel_btn_style())
        del_cfg_btn.clicked.connect(self._delete_img_filter_config)
        apply_single_btn = QPushButton("Preview with This Config")
        apply_single_btn.setStyleSheet(get_cancel_btn_style())
        apply_single_btn.clicked.connect(self._apply_single_config_to_preview)
        ic_btn_row.addWidget(save_cfg_btn)
        ic_btn_row.addWidget(load_cfg_btn)
        ic_btn_row.addWidget(del_cfg_btn)
        ic_btn_row.addWidget(apply_single_btn)
        ic.addLayout(ic_btn_row)

        self._cfg_status_lbl = QLabel("")
        self._cfg_status_lbl.setWordWrap(True)
        ic.addWidget(self._cfg_status_lbl)

        outer.addWidget(img_cfg_sec)

        # ── Section 4: Batch Dataset Operations ───────────────────────
        batch_sec = CollapsibleSection("Apply Filter to Dataset")
        dg = batch_sec.content_layout()
        dg.setSpacing(6)

        self._ds2_row, self.ds2_edit, ds2_btn = _dir_row("Dataset dir:", "Path with images/ and labels/")
        if default_dataset_dir:
            self.ds2_edit.setText(default_dataset_dir)
        ds2_btn.clicked.connect(lambda: _browse_dir(self, self.ds2_edit))
        dg.addWidget(self._ds2_row)

        self._out2_row, self.out2_edit, out2_btn = _dir_row("Output dir:", "Leave empty → dataset_filtered")
        out2_btn.clicked.connect(lambda: _browse_dir(self, self.out2_edit))
        dg.addWidget(self._out2_row)

        opts_row = QHBoxLayout()
        self.copy_orig2_cb = QCheckBox("Copy originals")
        self.copy_orig2_cb.setChecked(True)
        opts_row.addWidget(self.copy_orig2_cb)
        quality_row2, self.quality2_spin = _int_row("JPEG quality:", 1, 100, 100, 90)
        opts_row.addWidget(quality_row2)
        opts_row.addStretch()
        dg.addLayout(opts_row)

        mode_grp = QGroupBox("Which filter to apply")
        mg = QVBoxLayout(mode_grp)
        self._mode_m1 = QCheckBox("M1")
        self._mode_m2 = QCheckBox("M2")
        self._mode_m3 = QCheckBox("M3")
        self._mode_m2.setChecked(True)
        for cb in (self._mode_m1, self._mode_m2, self._mode_m3):
            mg.addWidget(cb)
        dg.addWidget(mode_grp)

        apply_btns = QHBoxLayout()
        apply_btns.addStretch()
        self.apply_btn = QPushButton("Apply Selected Filter(s)")
        self.apply_btn.setStyleSheet(get_ok_btn_style())
        self.apply_btn.clicked.connect(self._run_filter)
        apply_btns.addWidget(self.apply_btn)
        dg.addLayout(apply_btns)

        outer.addWidget(batch_sec)

        # ── Section 5: Random Filter Augmentation ─────────────────────
        rand_sec = CollapsibleSection("Random Filter Augmentation (range-based)", collapsed=True)
        rg = rand_sec.content_layout()
        rg.setSpacing(4)
        rg.addWidget(QLabel("Assign a random center and width per image within the ranges below:"))

        self._rc_min_row, self.rc_min_spin = _int_row("Center min:", 0, 255, 90)
        self._rc_max_row, self.rc_max_spin = _int_row("Center max:", 0, 255, 165)
        self._rw_min_row, self.rw_min_spin = _int_row("Width min:", 1, 510, 50)
        self._rw_max_row, self.rw_max_spin = _int_row("Width max:", 1, 510, 200)
        seed_row_r, self.seed2_spin = _int_row("Seed:", 0, 999999, 42)
        for w in (self._rc_min_row, self._rc_max_row, self._rw_min_row, self._rw_max_row, seed_row_r):
            rg.addWidget(w)

        rand_btns = QHBoxLayout()
        rand_btns.addStretch()
        self.rand_btn = QPushButton("Apply Randomly to Dataset")
        self.rand_btn.setStyleSheet(get_ok_btn_style())
        self.rand_btn.clicked.connect(self._run_random)
        rand_btns.addWidget(self.rand_btn)
        rg.addLayout(rand_btns)

        outer.addWidget(rand_sec)

        # ── Section 6: Mixed Preset Augmentation ──────────────────────
        mixed_sec = CollapsibleSection("Mixed Preset Augmentation", collapsed=False)
        mx = mixed_sec.content_layout()
        mx.setSpacing(6)

        mx.addWidget(QLabel(
            "Select a fraction of images from the dataset and apply a mix of named\n"
            "window presets according to the ratios below. CLAHE is applied randomly."
        ))

        frac_row, self.mix_frac_spin = _labeled_dspinbox(
            "Augmentation fraction:", 0.01, 1.0, 0.20, 0.01
        )
        mx.addWidget(frac_row)

        clahe_row, self.mix_clahe_spin = _labeled_dspinbox(
            "CLAHE probability:", 0.0, 1.0, 0.50, 0.05
        )
        mx.addWidget(clahe_row)

        mix_seed_row, self.mix_seed_spin = _int_row("Seed:", 0, 999999, 42)
        mx.addWidget(mix_seed_row)

        mix_copy_row = QHBoxLayout()
        self.mix_copy_orig_cb = QCheckBox("Copy originals to output")
        self.mix_copy_orig_cb.setChecked(True)
        self.mix_merge_cb = QCheckBox("Merge augmented images into source dataset")
        self.mix_merge_cb.setToolTip(
            "When checked, augmented images are written directly into the source\n"
            "dataset's images/ and labels/ folders (output dir is ignored)."
        )
        self.mix_merge_cb.toggled.connect(self._on_merge_toggle)
        mix_copy_row.addWidget(self.mix_copy_orig_cb)
        mix_copy_row.addWidget(self.mix_merge_cb)
        mix_copy_row.addStretch()
        mx.addLayout(mix_copy_row)

        # Preset table
        mx.addWidget(QLabel("Preset mix (each row: name, center, width, ratio):"))
        self.mix_table = QTableWidget(0, 4)
        self.mix_table.setHorizontalHeaderLabels(["Name", "Center", "Width", "Ratio"])
        self.mix_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.mix_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self.mix_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        self.mix_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        self.mix_table.verticalHeader().setVisible(False)
        self.mix_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.mix_table.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._mix_table_expanded = True
        mx.addWidget(self.mix_table)

        table_btn_row = QHBoxLayout()
        add_preset_btn = QPushButton("+ Add Preset")
        add_preset_btn.setStyleSheet(get_cancel_btn_style())
        add_preset_btn.clicked.connect(self._add_mix_preset_row)
        remove_preset_btn = QPushButton("Remove Selected")
        remove_preset_btn.setStyleSheet(get_cancel_btn_style())
        remove_preset_btn.clicked.connect(self._remove_mix_preset_row)
        reset_preset_btn = QPushButton("Reset to Defaults")
        reset_preset_btn.setStyleSheet(get_cancel_btn_style())
        reset_preset_btn.clicked.connect(self._reset_mix_presets)
        self._mix_table_toggle_btn = QPushButton("Compact view")
        self._mix_table_toggle_btn.setStyleSheet(get_cancel_btn_style())
        self._mix_table_toggle_btn.clicked.connect(self._toggle_mix_table_view)
        table_btn_row.addWidget(add_preset_btn)
        table_btn_row.addWidget(remove_preset_btn)
        table_btn_row.addWidget(reset_preset_btn)
        table_btn_row.addWidget(self._mix_table_toggle_btn)
        table_btn_row.addStretch()
        mx.addLayout(table_btn_row)

        # I/O for mixed aug
        self._mix_ds_row, self.mix_ds_edit, mix_ds_btn = _dir_row("Dataset dir:", "Source dataset with images/ and labels/")
        if default_dataset_dir:
            self.mix_ds_edit.setText(default_dataset_dir)
        mix_ds_btn.clicked.connect(lambda: _browse_dir(self, self.mix_ds_edit))
        mx.addWidget(self._mix_ds_row)

        self._mix_out_row, self.mix_out_edit, mix_out_btn = _dir_row("Output dir:", "Destination (ignored if merge checked)")
        mix_out_btn.clicked.connect(lambda: _browse_dir(self, self.mix_out_edit))
        mx.addWidget(self._mix_out_row)

        mix_quality_row, self.mix_quality_spin = _int_row("JPEG quality:", 1, 100, 100)
        mx.addWidget(mix_quality_row)

        mix_run_row = QHBoxLayout()
        mix_run_row.addStretch()
        self.mix_run_btn = QPushButton("Run Mixed Augmentation")
        self.mix_run_btn.setStyleSheet(get_ok_btn_style())
        self.mix_run_btn.clicked.connect(self._run_mixed)
        mix_run_row.addWidget(self.mix_run_btn)
        mx.addLayout(mix_run_row)

        outer.addWidget(mixed_sec)

        # Shared progress / status
        self.progress_bar2 = QProgressBar()
        self.progress_bar2.setRange(0, 100)
        self.progress_bar2.hide()
        outer.addWidget(self.progress_bar2)

        self.status2_label = QLabel("")
        self.status2_label.setWordWrap(True)
        outer.addWidget(self.status2_label)

        outer.addStretch()

        scroll.setWidget(inner)
        top = QVBoxLayout(self)
        top.setContentsMargins(0, 0, 0, 0)
        top.addWidget(scroll)

        # Populate default mix presets
        self._reset_mix_presets()

        # Load current image on init
        if current_image_path and Path(current_image_path).exists():
            self._load_image(current_image_path)

    # ── preview helpers ──────────────────────────────────────────────

    def _browse_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Image", self._img_path_edit.text(),
            "Images (*.jpg *.jpeg *.png *.bmp *.tiff)"
        )
        if path:
            self._img_path_edit.setText(path)
            self._load_image(path)

    def _load_preview_image(self):
        path = self._img_path_edit.text().strip()
        if path and Path(path).exists():
            self._load_image(path)

    def _load_image(self, path: str):
        img = imread_unicode(Path(path))
        if img is None:
            return
        self._bgr_image = img
        self._current_image_path = path
        self._img_path_edit.setText(path)
        self._refresh_preview()
        # Also auto-load saved config if present
        self._maybe_show_saved_config_hint(path)

    def _maybe_show_saved_config_hint(self, path: str):
        cfg = load_image_filter_config(path)
        if cfg:
            self._cfg_status_lbl.setText(
                f"Saved config for this image: C={cfg.get('center','?')}  "
                f"W={cfg.get('width','?')}  CLAHE={cfg.get('clahe', False)}"
            )
        else:
            self._cfg_status_lbl.setText("")

    def _apply_preset(self, center: int, width: int):
        self._preset_c_spins["M2"].setValue(center)
        self._preset_w_spins["M2"].setValue(width)
        self._mode_m2.setChecked(True)
        self._refresh_preview()

    def _refresh_preview(self):
        if self._bgr_image is None:
            return
        settings = [
            (self._preset_c_spins["M1"].value(), self._preset_w_spins["M1"].value()),
            (self._preset_c_spins["M2"].value(), self._preset_w_spins["M2"].value()),
            (self._preset_c_spins["M3"].value(), self._preset_w_spins["M3"].value()),
        ]
        self._preview.update_images(self._bgr_image, settings, self._clahe_cb.isChecked())

    def update_current_image(self, image_path: str):
        """Called by parent when the user opens a new image in the main window."""
        if image_path and Path(image_path).exists():
            self._load_image(image_path)

    # ── drag & drop image loading ────────────────────────────────────

    _IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")

    def _extract_dropped_image(self, mime) -> str:
        candidates = []
        if mime.hasUrls():
            candidates.extend(
                u.toLocalFile() for u in mime.urls() if u.isLocalFile()
            )
        if mime.hasText():
            candidates.append(mime.text().strip())
        for cand in candidates:
            if not cand:
                continue
            p = Path(cand)
            if p.exists() and p.suffix.lower() in self._IMAGE_EXTS:
                return str(p)
        return ""

    def dragEnterEvent(self, event):
        if self._extract_dropped_image(event.mimeData()):
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragMoveEvent(self, event):
        if self._extract_dropped_image(event.mimeData()):
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event):
        path = self._extract_dropped_image(event.mimeData())
        if path:
            self._load_image(path)
            event.acceptProposedAction()
        else:
            event.ignore()

    # ── per-image filter config ──────────────────────────────────────

    def _save_img_filter_config(self):
        path = self._current_image_path
        if not path or not Path(path).exists():
            QMessageBox.warning(self, "No Image", "Load an image first.")
            return
        config = {
            "center": self._single_c_spin.value(),
            "width": self._single_w_spin.value(),
            "clahe": self._single_clahe_cb.isChecked(),
        }
        save_image_filter_config(path, config)
        self._cfg_status_lbl.setText(
            f"Saved: C={config['center']}  W={config['width']}  CLAHE={config['clahe']}"
        )

    def _load_img_filter_config(self):
        path = self._current_image_path
        if not path or not Path(path).exists():
            QMessageBox.warning(self, "No Image", "Load an image first.")
            return
        cfg = load_image_filter_config(path)
        if not cfg:
            QMessageBox.information(self, "No Config", "No saved filter config found for this image.")
            return
        self._single_c_spin.setValue(cfg.get("center", 128))
        self._single_w_spin.setValue(cfg.get("width", 110))
        self._single_clahe_cb.setChecked(cfg.get("clahe", False))
        self._cfg_status_lbl.setText(
            f"Loaded: C={cfg.get('center','?')}  W={cfg.get('width','?')}  CLAHE={cfg.get('clahe', False)}"
        )

    def _delete_img_filter_config(self):
        path = self._current_image_path
        if not path:
            return
        delete_image_filter_config(path)
        self._cfg_status_lbl.setText("Config deleted.")

    def _apply_single_config_to_preview(self):
        if self._bgr_image is None:
            return
        c = self._single_c_spin.value()
        w = self._single_w_spin.value()
        use_clahe = self._single_clahe_cb.isChecked()
        # Temporarily override M2 values for preview
        self._preset_c_spins["M2"].blockSignals(True)
        self._preset_w_spins["M2"].blockSignals(True)
        self._preset_c_spins["M2"].setValue(c)
        self._preset_w_spins["M2"].setValue(w)
        self._preset_c_spins["M2"].blockSignals(False)
        self._preset_w_spins["M2"].blockSignals(False)
        self._clahe_cb.blockSignals(True)
        self._clahe_cb.setChecked(use_clahe)
        self._clahe_cb.blockSignals(False)
        self._refresh_preview()

    # ── dataset filter ops ───────────────────────────────────────────

    def _get_presets_to_apply(self):
        presets = []
        for name, cb in [("M1", self._mode_m1), ("M2", self._mode_m2), ("M3", self._mode_m3)]:
            if cb.isChecked():
                presets.append((name, self._preset_c_spins[name].value(), self._preset_w_spins[name].value()))
        return presets

    def _run_filter(self):
        ds = self.ds2_edit.text().strip()
        presets = self._get_presets_to_apply()
        if not ds:
            QMessageBox.warning(self, "Missing Path", "Set a dataset directory.")
            return
        if not Path(ds, "images").exists():
            QMessageBox.warning(self, "Invalid Dataset", f"No 'images' subfolder in:\n{ds}")
            return
        if not presets:
            QMessageBox.warning(self, "No Filter", "Select at least one filter (M1/M2/M3).")
            return

        out = self.out2_edit.text().strip()
        if not out:
            out = str(Path(ds).parent / (Path(ds).name + "_filtered"))

        self.apply_btn.setEnabled(False)
        self.rand_btn.setEnabled(False)
        self.mix_run_btn.setEnabled(False)
        self.progress_bar2.show()
        self.status2_label.setText("Running…")

        self._pending_presets = list(presets)
        self._multi_preset_run = len(presets) > 1
        self._filter_out_base = out
        self._copy_orig_flag = self.copy_orig2_cb.isChecked()
        self._quality_flag = self.quality2_spin.value()
        self._filter_ds = ds
        self._run_next_preset()

    def _run_next_preset(self):
        if not self._pending_presets:
            self._enable_buttons()
            self.progress_bar2.hide()
            QMessageBox.information(self, "Complete", self.status2_label.text())
            return

        name, c, w = self._pending_presets.pop(0)
        if self._multi_preset_run:
            out_dir = str(Path(self._filter_out_base) / name)
        else:
            out_dir = self._filter_out_base

        params = {
            "dataset_dir": self._filter_ds,
            "output_dir": out_dir,
            "center": c,
            "width": w,
            "use_clahe": self._clahe_cb.isChecked(),
            "copy_originals": self._copy_orig_flag,
            "output_jpeg_quality": self._quality_flag,
        }

        self._worker = _FilterWorker(params)
        self._worker.progress.connect(self.progress_bar2.setValue)
        self._worker.finished.connect(self._on_filter_done)
        self._worker.start()

    def _on_filter_done(self, result: dict):
        if "error" in result:
            self._enable_buttons()
            self.progress_bar2.hide()
            self.status2_label.setText(f"Error: {result['error']}")
            QMessageBox.critical(self, "Error", result["error"])
            return
        msg = (
            f"Filter applied – originals: {result['originals']} | "
            f"filtered: {result.get('filtered', 0)} | "
            f"output: {result['output_dir']}"
        )
        self.status2_label.setText(msg)
        self._run_next_preset()

    def _run_random(self):
        ds = self.ds2_edit.text().strip()
        if not ds:
            QMessageBox.warning(self, "Missing Path", "Set a dataset directory.")
            return
        if not Path(ds, "images").exists():
            QMessageBox.warning(self, "Invalid Dataset", f"No 'images' subfolder in:\n{ds}")
            return

        out = self.out2_edit.text().strip()
        if not out:
            out = str(Path(ds).parent / (Path(ds).name + "_rand_filtered"))

        c_min = self.rc_min_spin.value()
        c_max = self.rc_max_spin.value()
        w_min = self.rw_min_spin.value()
        w_max = self.rw_max_spin.value()

        if c_min > c_max or w_min > w_max:
            QMessageBox.warning(self, "Invalid Range", "Min values must be ≤ max values.")
            return

        params = {
            "dataset_dir": ds,
            "output_dir": out,
            "center": (c_min + c_max) // 2,
            "width": (w_min + w_max) // 2,
            "use_clahe": self._clahe_cb.isChecked(),
            "copy_originals": self.copy_orig2_cb.isChecked(),
            "output_jpeg_quality": self.quality2_spin.value(),
            "randomize": True,
            "center_range": (c_min, c_max),
            "width_range": (w_min, w_max),
            "seed": self.seed2_spin.value(),
        }

        self.apply_btn.setEnabled(False)
        self.rand_btn.setEnabled(False)
        self.mix_run_btn.setEnabled(False)
        self.progress_bar2.show()
        self.status2_label.setText("Running random filter…")

        self._worker = _FilterWorker(params)
        self._worker.progress.connect(self.progress_bar2.setValue)
        self._worker.finished.connect(self._on_rand_done)
        self._worker.start()

    def _on_rand_done(self, result: dict):
        self._enable_buttons()
        self.progress_bar2.hide()
        if "error" in result:
            self.status2_label.setText(f"Error: {result['error']}")
            QMessageBox.critical(self, "Error", result["error"])
        else:
            msg = (
                f"Done! Originals: {result['originals']} | "
                f"Filtered: {result.get('filtered', 0)} | "
                f"Total: {result['total']}\n"
                f"Output: {result['output_dir']}"
            )
            self.status2_label.setText(msg)
            QMessageBox.information(self, "Complete", msg)

    # ── mixed augmentation ───────────────────────────────────────────

    def _on_merge_toggle(self, checked: bool):
        self.mix_out_edit.setEnabled(not checked)
        self.mix_copy_orig_cb.setEnabled(not checked)

    def _add_mix_preset_row(self, name="", center=128, width=110, ratio=0.5):
        row = self.mix_table.rowCount()
        self.mix_table.insertRow(row)
        self.mix_table.setItem(row, 0, QTableWidgetItem(str(name)))
        c_spin = QSpinBox()
        c_spin.setRange(0, 255)
        c_spin.setValue(int(center))
        self.mix_table.setCellWidget(row, 1, c_spin)
        w_spin = QSpinBox()
        w_spin.setRange(1, 510)
        w_spin.setValue(int(width))
        self.mix_table.setCellWidget(row, 2, w_spin)
        r_spin = QDoubleSpinBox()
        r_spin.setRange(0.01, 10.0)
        r_spin.setSingleStep(0.05)
        r_spin.setDecimals(2)
        r_spin.setValue(float(ratio))
        self.mix_table.setCellWidget(row, 3, r_spin)
        self._update_mix_table_height()

    def _remove_mix_preset_row(self):
        row = self.mix_table.currentRow()
        if row >= 0:
            self.mix_table.removeRow(row)
        self._update_mix_table_height()

    def _reset_mix_presets(self):
        self.mix_table.setRowCount(0)
        self._add_mix_preset_row("M2", M2_PRESET[0], M2_PRESET[1], 0.5)
        self._add_mix_preset_row("M3", M3_PRESET[0], M3_PRESET[1], 0.5)

    def _toggle_mix_table_view(self):
        self._mix_table_expanded = not self._mix_table_expanded
        self._mix_table_toggle_btn.setText(
            "Compact view" if self._mix_table_expanded else "Show all rows"
        )
        self._update_mix_table_height()

    def _update_mix_table_height(self):
        """Size the preset table so every row is visible (expanded) or
        keep a compact, scrollable height."""
        rows = self.mix_table.rowCount()
        header_h = self.mix_table.horizontalHeader().height()
        row_h = self.mix_table.verticalHeader().defaultSectionSize()
        if self._mix_table_expanded:
            total = header_h + row_h * max(rows, 1) + 4
            self.mix_table.setMinimumHeight(total)
            self.mix_table.setMaximumHeight(total)
        else:
            compact = header_h + row_h * min(max(rows, 1), 3) + 4
            self.mix_table.setMinimumHeight(compact)
            self.mix_table.setMaximumHeight(compact)

    def _collect_mix_presets(self):
        presets = []
        for row in range(self.mix_table.rowCount()):
            name_item = self.mix_table.item(row, 0)
            name = name_item.text().strip() if name_item else f"P{row}"
            c_spin = self.mix_table.cellWidget(row, 1)
            w_spin = self.mix_table.cellWidget(row, 2)
            r_spin = self.mix_table.cellWidget(row, 3)
            presets.append({
                "name": name or f"P{row}",
                "center": c_spin.value() if c_spin else 128,
                "width": w_spin.value() if w_spin else 110,
                "ratio": r_spin.value() if r_spin else 1.0,
            })
        return presets

    def _run_mixed(self):
        ds = self.mix_ds_edit.text().strip()
        if not ds:
            QMessageBox.warning(self, "Missing Path", "Set a dataset directory.")
            return
        if not Path(ds, "images").exists():
            QMessageBox.warning(self, "Invalid Dataset", f"No 'images' subfolder in:\n{ds}")
            return

        presets = self._collect_mix_presets()
        if not presets:
            QMessageBox.warning(self, "No Presets", "Add at least one preset row.")
            return

        merge = self.mix_merge_cb.isChecked()
        out = self.mix_out_edit.text().strip()
        if not merge and not out:
            out = str(Path(ds).parent / (Path(ds).name + "_mixed_aug"))

        params = {
            "dataset_dir": ds,
            "output_dir": out if not merge else ds,
            "augmentation_fraction": self.mix_frac_spin.value(),
            "presets": presets,
            "clahe_probability": self.mix_clahe_spin.value(),
            "copy_originals": self.mix_copy_orig_cb.isChecked() and not merge,
            "merge_into_source": merge,
            "output_jpeg_quality": self.mix_quality_spin.value(),
            "seed": self.mix_seed_spin.value(),
        }

        self.apply_btn.setEnabled(False)
        self.rand_btn.setEnabled(False)
        self.mix_run_btn.setEnabled(False)
        self.progress_bar2.setValue(0)
        self.progress_bar2.show()
        self.status2_label.setText("Running mixed augmentation…")

        self._worker = _MixedAugWorker(params)
        self._worker.progress.connect(self.progress_bar2.setValue)
        self._worker.finished.connect(self._on_mixed_done)
        self._worker.start()

    def _on_mixed_done(self, result: dict):
        self._enable_buttons()
        self.progress_bar2.hide()
        if "error" in result:
            self.status2_label.setText(f"Error: {result['error']}")
            QMessageBox.critical(self, "Error", result["error"])
        else:
            msg = (
                f"Done! Originals copied: {result['originals']} | "
                f"Augmented: {result['augmented']} | "
                f"Total: {result['total']}\n"
                f"Output: {result['output_dir']}"
            )
            self.status2_label.setText(msg)
            QMessageBox.information(self, "Complete", msg)

    def _enable_buttons(self):
        self.apply_btn.setEnabled(True)
        self.rand_btn.setEnabled(True)
        self.mix_run_btn.setEnabled(True)


# ── Main dialog ────────────────────────────────────────────────────────

class AugmentationDialog(QDialog):
    """Main data augmentation dialog with Random Crop and Window Filter tabs."""

    def __init__(self, current_image_path: str = "", dataset_dir: str = "", parent=None):
        super().__init__(parent)
        self.setWindowTitle("Data Augmentation")
        self.setMinimumSize(750, 700)
        self.setWindowFlags(
            self.windowFlags() & ~Qt.WindowType.WindowContextHelpButtonHint
        )
        self.setStyleSheet(get_dialog_style())
        self.setAcceptDrops(True)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)

        self._tabs = QTabWidget()
        self._crop_tab = _RandomCropTab(default_dataset_dir=dataset_dir, parent=self)
        self._filter_tab = _FilterTab(
            current_image_path=current_image_path,
            default_dataset_dir=dataset_dir,
            parent=self,
        )
        self._tabs.addTab(self._crop_tab, "Random Crop")
        self._tabs.addTab(self._filter_tab, "Window Filter")
        layout.addWidget(self._tabs)

        close_row = QHBoxLayout()
        close_row.addStretch()
        close_btn = QPushButton("Close")
        close_btn.setStyleSheet(get_cancel_btn_style())
        close_btn.clicked.connect(self.close)
        close_row.addWidget(close_btn)
        layout.addLayout(close_row)

    def dragEnterEvent(self, event):
        if self._filter_tab._extract_dropped_image(event.mimeData()):
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragMoveEvent(self, event):
        if self._filter_tab._extract_dropped_image(event.mimeData()):
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event):
        path = self._filter_tab._extract_dropped_image(event.mimeData())
        if path:
            self._tabs.setCurrentWidget(self._filter_tab)
            self._filter_tab._load_image(path)
            event.acceptProposedAction()
        else:
            event.ignore()

    def update_current_image(self, image_path: str):
        """Propagate the currently open image to the filter preview."""
        self._filter_tab.update_current_image(image_path)

    def set_dataset_dir(self, directory: str):
        """Pre-fill dataset directory fields when a project/folder is opened."""
        if directory:
            self._crop_tab.ds_edit.setText(directory)
            self._filter_tab.ds2_edit.setText(directory)
            self._filter_tab.mix_ds_edit.setText(directory)
