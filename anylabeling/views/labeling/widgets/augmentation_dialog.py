"""Data augmentation dialog – Random Crop and Window Filter tabs."""

import os
from pathlib import Path

import cv2
import numpy as np
from PyQt6 import QtCore, QtGui, QtWidgets
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtWidgets import (
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
    QRadioButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
    QButtonGroup,
    QSizePolicy,
)

from anylabeling.views.labeling.utils.augmentation_engine import (
    M1_PRESET,
    M2_PRESET,
    M3_PRESET,
    apply_filter,
    imread_unicode,
    run_filter_dataset,
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
        result = run_random_crop(**self._params, progress_callback=self.progress.emit)
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


# ── Helpers ────────────────────────────────────────────────────────────

def _labeled_slider(label_text: str, lo: int, hi: int, default: int, step: int = 1):
    """Return (widget, slider, value_label) for a row with a slider."""
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


def _dir_row(label_text: str, placeholder: str = ""):
    """Return (widget, line_edit, browse_btn) for a directory picker row."""
    row = QWidget()
    lay = QHBoxLayout(row)
    lay.setContentsMargins(0, 0, 0, 0)
    lbl = QLabel(label_text)
    lbl.setFixedWidth(100)
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
        """settings: list of (center, width) for M1/M2/M3; original shown as-is."""
        images = [bgr_original]
        for c, w in settings:
            images.append(apply_filter(bgr_original, c, w, use_clahe))

        for label, img in zip(self._labels, images):
            self._set_pixmap(label, img)

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
        outer = QVBoxLayout(self)
        outer.setContentsMargins(10, 10, 10, 10)
        outer.setSpacing(10)

        # Parameters
        params_group = QGroupBox("Augmentation Parameters")
        pg = QVBoxLayout(params_group)
        pg.setSpacing(6)

        self._n_row, self.n_spin, _ = self._int_row("Augmentations per image:", 1, 50, 1)
        pg.addWidget(self._n_row)

        self._minr_row, self.min_ratio_spin = _labeled_dspinbox("Min crop ratio:", 0.05, 0.95, 0.30)
        pg.addWidget(self._minr_row)

        self._maxr_row, self.max_ratio_spin = _labeled_dspinbox("Max crop ratio:", 0.05, 0.95, 0.70)
        pg.addWidget(self._maxr_row)

        self._mmask_row, self.min_mask_spin = _labeled_dspinbox("Min mask ratio:", 0.01, 0.99, 0.10)
        pg.addWidget(self._mmask_row)

        quality_row, self.quality_spin, _ = self._int_row("JPEG quality (0-100):", 1, 100, 100)
        pg.addWidget(quality_row)

        seed_row, self.seed_spin, _ = self._int_row("Random seed:", 0, 999999, 42)
        pg.addWidget(seed_row)

        self.copy_orig_cb = QCheckBox("Copy originals to output")
        self.copy_orig_cb.setChecked(True)
        pg.addWidget(self.copy_orig_cb)

        outer.addWidget(params_group)

        # I/O
        io_group = QGroupBox("Directories")
        ig = QVBoxLayout(io_group)

        self._ds_row, self.ds_edit, ds_btn = _dir_row("Dataset dir:", "Path to folder with images/ and labels/")
        if default_dataset_dir:
            self.ds_edit.setText(default_dataset_dir)
        ds_btn.clicked.connect(lambda: self._browse(self.ds_edit))
        ig.addWidget(self._ds_row)

        self._out_row, self.out_edit, out_btn = _dir_row("Output dir:", "Destination folder (will be created)")
        out_btn.clicked.connect(lambda: self._browse(self.out_edit))
        ig.addWidget(self._out_row)

        outer.addWidget(io_group)

        # Progress
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
        self.run_btn.clicked.connect(self._run)
        self._worker = None

    @staticmethod
    def _int_row(label_text: str, lo: int, hi: int, default: int):
        row = QWidget()
        lay = QHBoxLayout(row)
        lay.setContentsMargins(0, 0, 0, 0)
        lbl = QLabel(label_text)
        lbl.setFixedWidth(175)
        spin = QSpinBox()
        spin.setRange(lo, hi)
        spin.setValue(default)
        lay.addWidget(lbl)
        lay.addWidget(spin)
        return row, spin, lbl

    def _browse(self, edit: QLineEdit):
        d = QFileDialog.getExistingDirectory(self, "Select Directory", edit.text() or "")
        if d:
            edit.setText(d)

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
            msg = (
                f"Done! Originals: {result['originals']} | "
                f"Augmented: {result['augmented']} | "
                f"Total: {result['total']}\n"
                f"Output: {result['output_dir']}"
            )
            self.status_label.setText(msg)
            QMessageBox.information(self, "Complete", msg)


# ── Tab 2: Window Filter ───────────────────────────────────────────────

class _FilterTab(QWidget):
    def __init__(self, current_image_path: str = "", default_dataset_dir: str = "", parent=None):
        super().__init__(parent)
        self._current_image_path = current_image_path
        self._bgr_image = None
        self._worker = None

        outer = QVBoxLayout(self)
        outer.setContentsMargins(10, 10, 10, 10)
        outer.setSpacing(10)

        # --- Preview section ---
        preview_group = QGroupBox("Preview (current image)")
        prev_lay = QVBoxLayout(preview_group)

        self._preview = _PreviewPanel()
        scroll = QScrollArea()
        scroll.setWidget(self._preview)
        scroll.setWidgetResizable(True)
        scroll.setMinimumHeight(450)
        prev_lay.addWidget(scroll)

        load_row = QHBoxLayout()
        self._img_path_edit = QLineEdit(current_image_path)
        self._img_path_edit.setPlaceholderText("Image path for preview…")
        load_btn = QPushButton("Load")
        load_btn.setFixedWidth(65)
        load_btn.setStyleSheet(get_cancel_btn_style())
        load_btn.clicked.connect(self._load_preview_image)
        load_row.addWidget(QLabel("Image:"))
        load_row.addWidget(self._img_path_edit)
        load_row.addWidget(load_btn)
        browse_img_btn = QPushButton("Browse…")
        browse_img_btn.setFixedWidth(75)
        browse_img_btn.setStyleSheet(get_cancel_btn_style())
        browse_img_btn.clicked.connect(self._browse_image)
        load_row.addWidget(browse_img_btn)
        prev_lay.addLayout(load_row)

        outer.addWidget(preview_group)

        # --- Filter settings ---
        settings_group = QGroupBox("Filter Settings")
        sg = QVBoxLayout(settings_group)
        sg.setSpacing(6)

        self._clahe_cb = QCheckBox("Apply CLAHE on top of window filter")
        self._clahe_cb.toggled.connect(self._refresh_preview)
        sg.addWidget(self._clahe_cb)

        # M1/M2/M3 preset buttons row
        preset_row = QHBoxLayout()
        preset_row.addWidget(QLabel("Presets:"))
        for name, (c, w) in [("M1", M1_PRESET), ("M2", M2_PRESET), ("M3", M3_PRESET)]:
            btn = QPushButton(f"{name}  (C={c}, W={w})")
            btn.setStyleSheet(get_cancel_btn_style())
            btn.clicked.connect(lambda checked, _c=c, _w=w: self._apply_preset(_c, _w))
            preset_row.addWidget(btn)
        preset_row.addStretch()
        sg.addLayout(preset_row)

        # Preset editor
        preset_edit_group = QGroupBox("Customize Presets (used in preview & dataset ops)")
        peg = QtWidgets.QGridLayout(preset_edit_group)
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

        sg.addWidget(preset_edit_group)
        outer.addWidget(settings_group)

        # --- Dataset operation ---
        ds_group = QGroupBox("Apply to Dataset")
        dg = QVBoxLayout(ds_group)
        dg.setSpacing(6)

        self._ds2_row, self.ds2_edit, ds2_btn = _dir_row("Dataset dir:", "Path with images/ and labels/")
        if default_dataset_dir:
            self.ds2_edit.setText(default_dataset_dir)
        ds2_btn.clicked.connect(lambda: self._browse_dir(self.ds2_edit))
        dg.addWidget(self._ds2_row)

        self._out2_row, self.out2_edit, out2_btn = _dir_row("Output dir:", "Leave empty to use dataset_dir + '_filtered'")
        out2_btn.clicked.connect(lambda: self._browse_dir(self.out2_edit))
        dg.addWidget(self._out2_row)

        self.copy_orig2_cb = QCheckBox("Copy originals to output")
        self.copy_orig2_cb.setChecked(True)
        dg.addWidget(self.copy_orig2_cb)

        quality_row2, self.quality2_spin, _ = _RandomCropTab._int_row("JPEG quality:", 1, 100, 100)
        dg.addWidget(quality_row2)

        # Mode: which preset(s) to apply
        mode_group = QGroupBox("Which filter to apply")
        mg = QVBoxLayout(mode_group)
        self._mode_m1 = QCheckBox("M1")
        self._mode_m2 = QCheckBox("M2")
        self._mode_m3 = QCheckBox("M3")
        self._mode_m2.setChecked(True)
        for cb in (self._mode_m1, self._mode_m2, self._mode_m3):
            mg.addWidget(cb)
        dg.addWidget(mode_group)

        apply_btns = QHBoxLayout()
        apply_btns.addStretch()
        self.apply_btn = QPushButton("Apply Selected Filter(s)")
        self.apply_btn.setStyleSheet(get_ok_btn_style())
        self.apply_btn.clicked.connect(self._run_filter)
        apply_btns.addWidget(self.apply_btn)
        dg.addLayout(apply_btns)

        # Random filter augmentation sub-section
        rand_group = QGroupBox("Random Filter Augmentation")
        rg = QVBoxLayout(rand_group)
        rg.setSpacing(4)
        rg.addWidget(QLabel("Randomize center and width per image within ranges:"))

        self._rc_min_row, self.rc_min_spin, _ = _RandomCropTab._int_row("Center min:", 0, 255, 90)
        self._rc_max_row, self.rc_max_spin, _ = _RandomCropTab._int_row("Center max:", 0, 255, 165)
        self._rw_min_row, self.rw_min_spin, _ = _RandomCropTab._int_row("Width min:", 1, 510, 50)
        self._rw_max_row, self.rw_max_spin, _ = _RandomCropTab._int_row("Width max:", 1, 510, 200)
        seed_row_r, self.seed2_spin, _ = _RandomCropTab._int_row("Seed:", 0, 999999, 42)
        for w in (self._rc_min_row, self._rc_max_row, self._rw_min_row, self._rw_max_row, seed_row_r):
            rg.addWidget(w)

        rand_btns = QHBoxLayout()
        rand_btns.addStretch()
        self.rand_btn = QPushButton("Apply Randomly to Dataset")
        self.rand_btn.setStyleSheet(get_ok_btn_style())
        self.rand_btn.clicked.connect(self._run_random)
        rand_btns.addWidget(self.rand_btn)
        rg.addLayout(rand_btns)

        dg.addWidget(rand_group)
        outer.addWidget(ds_group)

        self.progress_bar2 = QProgressBar()
        self.progress_bar2.setRange(0, 100)
        self.progress_bar2.hide()
        outer.addWidget(self.progress_bar2)

        self.status2_label = QLabel("")
        self.status2_label.setWordWrap(True)
        outer.addWidget(self.status2_label)

        outer.addStretch()

        # Load current image on init
        if current_image_path and Path(current_image_path).exists():
            self._load_image(current_image_path)

    # -- preview helpers --

    def _browse_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Image", self._img_path_edit.text(),
            "Images (*.jpg *.jpeg *.png *.bmp *.tiff)"
        )
        if path:
            self._img_path_edit.setText(path)
            self._load_image(path)

    def _browse_dir(self, edit: QLineEdit):
        d = QFileDialog.getExistingDirectory(self, "Select Directory", edit.text() or "")
        if d:
            edit.setText(d)

    def _load_preview_image(self):
        path = self._img_path_edit.text().strip()
        if path and Path(path).exists():
            self._load_image(path)

    def _load_image(self, path: str):
        img = imread_unicode(Path(path))
        if img is None:
            return
        self._bgr_image = img
        self._refresh_preview()

    def _apply_preset(self, center: int, width: int):
        """Set the M2 spinboxes to the clicked preset values and refresh."""
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

    # -- dataset operations --

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

        # Run each selected preset sequentially (use same output dir for all)
        self.apply_btn.setEnabled(False)
        self.rand_btn.setEnabled(False)
        self.progress_bar2.show()
        self.status2_label.setText("Running…")

        # Build params for the first preset; chain results via signals
        self._pending_presets = list(presets)
        self._multi_preset_run = len(presets) > 1
        self._filter_out_base = out
        self._copy_orig_flag = self.copy_orig2_cb.isChecked()
        self._quality_flag = self.quality2_spin.value()
        self._filter_ds = ds
        self._run_next_preset()

    def _run_next_preset(self):
        if not self._pending_presets:
            self.apply_btn.setEnabled(True)
            self.rand_btn.setEnabled(True)
            self.progress_bar2.hide()
            QMessageBox.information(self, "Complete", self.status2_label.text())
            return

        name, c, w = self._pending_presets.pop(0)
        # Use per-name subfolders when multiple presets are selected
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
            self.apply_btn.setEnabled(True)
            self.rand_btn.setEnabled(True)
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
        self.progress_bar2.show()
        self.status2_label.setText("Running random filter…")

        self._worker = _FilterWorker(params)
        self._worker.progress.connect(self.progress_bar2.setValue)
        self._worker.finished.connect(self._on_rand_done)
        self._worker.start()

    def _on_rand_done(self, result: dict):
        self.apply_btn.setEnabled(True)
        self.rand_btn.setEnabled(True)
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

    def update_current_image(self, image_path: str):
        """Called by parent when the user opens a new image."""
        self._img_path_edit.setText(image_path)
        self._current_image_path = image_path
        if image_path and Path(image_path).exists():
            self._load_image(image_path)


# ── Main dialog ────────────────────────────────────────────────────────

class AugmentationDialog(QDialog):
    """Main data augmentation dialog with Random Crop and Window Filter tabs."""

    def __init__(self, current_image_path: str = "", dataset_dir: str = "", parent=None):
        super().__init__(parent)
        self.setWindowTitle("Data Augmentation")
        self.setMinimumSize(700, 680)
        self.setWindowFlags(
            self.windowFlags() & ~Qt.WindowType.WindowContextHelpButtonHint
        )
        self.setStyleSheet(get_dialog_style())

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

    def update_current_image(self, image_path: str):
        """Propagate the currently open image to the filter preview."""
        self._filter_tab.update_current_image(image_path)

    def set_dataset_dir(self, directory: str):
        """Pre-fill dataset directory fields when a project/folder is opened."""
        if directory:
            self._crop_tab.ds_edit.setText(directory)
            self._filter_tab.ds2_edit.setText(directory)
