"""Backend logic for data augmentation: random crop and window filter."""

import shutil
from pathlib import Path

import cv2
import numpy as np

# Window filter presets (center, width) on 0-255 scale
M1_PRESET = (128, 50)   # Narrow/high-contrast – fine defects
M2_PRESET = (128, 110)  # Standard inspection
M3_PRESET = (128, 200)  # Wide range – broad features


# ── Image I/O ─────────────────────────────────────────────────────────

def imread_unicode(path: Path) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def imwrite_unicode(path: Path, image: np.ndarray, quality: int = 100) -> bool:
    ext = path.suffix.lower() if path.suffix else ".jpg"
    params = [cv2.IMWRITE_JPEG_QUALITY, quality] if ext in (".jpg", ".jpeg") else []
    ok, buf = cv2.imencode(ext, image, params)
    if not ok:
        return False
    buf.tofile(str(path))
    return True


# ── Window center/width filter ─────────────────────────────────────────

def apply_window(image: np.ndarray, center: int, width: int) -> np.ndarray:
    """Map pixel values through a window (center±width/2) → 0-255."""
    width = max(width, 1)
    low = center - width / 2.0
    high = center + width / 2.0
    windowed = np.clip((image.astype(np.float32) - low) / (high - low) * 255.0, 0.0, 255.0)
    return windowed.astype(np.uint8)


def apply_clahe(image: np.ndarray, clip_limit: float = 3.0,
                tile_grid: tuple = (8, 8)) -> np.ndarray:
    """Apply CLAHE to a BGR or grayscale image."""
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
    if len(image.shape) == 2:
        return clahe.apply(image)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def apply_filter(image: np.ndarray, center: int, width: int,
                 use_clahe: bool = False) -> np.ndarray:
    result = apply_window(image, center, width)
    if use_clahe:
        result = apply_clahe(result)
    return result


# ── YOLO segmentation I/O ─────────────────────────────────────────────

def read_yolo_seg(label_path: Path):
    """Returns list of (class_id, [(x_norm, y_norm), ...])."""
    annotations = []
    if not label_path.exists():
        return annotations
    for line in label_path.read_text(encoding="utf-8").strip().splitlines():
        parts = line.strip().split()
        if len(parts) < 7:
            continue
        cls = int(parts[0])
        coords = list(map(float, parts[1:]))
        pts = list(zip(coords[0::2], coords[1::2]))
        if pts and pts[0] == pts[-1] and len(pts) > 1:
            pts = pts[:-1]
        annotations.append((cls, pts))
    return annotations


def write_yolo_seg(label_path: Path, annotations):
    with open(label_path, "w", encoding="utf-8") as f:
        for cls_id, pts in annotations:
            coords = " ".join(f"{x:.6f} {y:.6f}" for x, y in pts)
            f.write(f"{cls_id} {coords}\n")


def poly_to_mask(pts, h: int, w: int) -> np.ndarray:
    arr = np.array([[round(x * w), round(y * h)] for x, y in pts], dtype=np.int32)
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [arr], 1)
    return mask


def mask_to_poly(mask: np.ndarray):
    h, w = mask.shape
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    if len(contour) < 3:
        return None
    return [(float(p[0][0]) / w, float(p[0][1]) / h) for p in contour]


# ── Random crop augmentation ──────────────────────────────────────────

def _crop_transform(img_h, img_w, crop_min, crop_max):
    try:
        import albumentations as A
    except ImportError:
        raise ImportError(
            "albumentations is required. Install with: pip install albumentations"
        )
    min_h = max(1, int(img_h * crop_min))
    max_h = max(min_h, int(img_h * crop_max))
    min_w = max(1, int(img_w * crop_min))
    max_w = max(min_w, int(img_w * crop_max))
    crop_h = int(np.random.randint(min_h, max_h + 1))
    crop_w = int(np.random.randint(min_w, max_w + 1))
    return A.Compose([A.RandomCrop(height=crop_h, width=crop_w, p=1.0)])


def _collect_images(img_dir: Path):
    paths = []
    for ext in ("*.jpg", "*.jpeg", "*.png"):
        paths.extend(sorted(img_dir.glob(ext)))
    return paths


def run_random_crop(
    dataset_dir: str,
    output_dir: str,
    n_aug_per_image: int = 1,
    crop_min_ratio: float = 0.30,
    crop_max_ratio: float = 0.70,
    min_mask_ratio: float = 0.10,
    copy_originals: bool = True,
    output_jpeg_quality: int = 100,
    seed: int = 42,
    progress_callback=None,
) -> dict:
    """Run random-crop augmentation on a YOLO-seg dataset folder.

    Returns a result dict with keys: originals, augmented, total, output_dir, error.
    """
    try:
        import albumentations  # noqa – just probe availability
    except ImportError:
        return {"error": "albumentations not installed. Run: pip install albumentations"}

    np.random.seed(seed)

    ds = Path(dataset_dir)
    out = Path(output_dir)
    img_in = ds / "images"
    lbl_in = ds / "labels"
    img_out = out / "images"
    lbl_out = out / "labels"

    for d in (img_out, lbl_out):
        d.mkdir(parents=True, exist_ok=True)

    image_paths = _collect_images(img_in)
    if not image_paths:
        return {"error": f"No images found in {img_in}"}

    total = len(image_paths)
    orig_copied = 0
    aug_saved = 0

    def _report(pct):
        if progress_callback:
            progress_callback(int(pct))

    if copy_originals:
        for i, p in enumerate(image_paths):
            shutil.copy2(p, img_out / p.name)
            lp = lbl_in / (p.stem + ".txt")
            if lp.exists():
                shutil.copy2(lp, lbl_out / lp.name)
            orig_copied += 1
            _report(i / total * 30)

    for i, img_path in enumerate(image_paths):
        img_bgr = imread_unicode(img_path)
        if img_bgr is None:
            continue
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        h, w = img_rgb.shape[:2]

        lbl_path = lbl_in / (img_path.stem + ".txt")
        annotations = read_yolo_seg(lbl_path)
        class_ids = [c for c, _ in annotations]
        orig_masks = [poly_to_mask(pts, h, w) for _, pts in annotations]
        orig_areas = [int(m.sum()) for m in orig_masks]

        for aug_i in range(n_aug_per_image):
            try:
                transform = _crop_transform(h, w, crop_min_ratio, crop_max_ratio)
                result = transform(image=img_rgb, masks=orig_masks)
            except Exception:
                continue

            aug_img = result["image"]
            aug_masks = result.get("masks", [])

            new_anns = []
            for cls_id, aug_mask, orig_area in zip(class_ids, aug_masks, orig_areas):
                if orig_area == 0:
                    continue
                if aug_mask.sum() / orig_area < min_mask_ratio:
                    continue
                pts = mask_to_poly(aug_mask)
                if pts is None or len(pts) < 3:
                    continue
                new_anns.append((cls_id, pts))

            out_stem = f"{img_path.stem}_crop{aug_i + 1}"
            imwrite_unicode(
                img_out / f"{out_stem}.jpg",
                cv2.cvtColor(aug_img, cv2.COLOR_RGB2BGR),
                output_jpeg_quality,
            )
            write_yolo_seg(lbl_out / f"{out_stem}.txt", new_anns)
            aug_saved += 1

        _report(30 + i / total * 70)

    for fname in ("data.yaml", "classes.txt"):
        src = ds / fname
        if src.exists():
            shutil.copy2(src, out / fname)

    _report(100)
    return {
        "originals": orig_copied,
        "augmented": aug_saved,
        "total": orig_copied + aug_saved,
        "output_dir": str(out),
    }


# ── Window filter dataset processing ──────────────────────────────────

def run_filter_dataset(
    dataset_dir: str,
    output_dir: str,
    center: int,
    width: int,
    use_clahe: bool = False,
    copy_originals: bool = True,
    output_jpeg_quality: int = 100,
    randomize: bool = False,
    center_range: tuple = (90, 165),
    width_range: tuple = (50, 200),
    seed: int = 42,
    progress_callback=None,
) -> dict:
    """Apply window filter to every image in a YOLO-seg dataset.

    When randomize=True each image gets a random center/width within the given ranges.
    Labels are copied as-is (the filter changes only pixel values, not annotations).
    """
    np.random.seed(seed)

    ds = Path(dataset_dir)
    out = Path(output_dir)
    img_in = ds / "images"
    lbl_in = ds / "labels"
    img_out = out / "images"
    lbl_out = out / "labels"

    for d in (img_out, lbl_out):
        d.mkdir(parents=True, exist_ok=True)

    image_paths = _collect_images(img_in)
    if not image_paths:
        return {"error": f"No images found in {img_in}"}

    total = len(image_paths)
    orig_copied = 0
    filtered_saved = 0

    if copy_originals:
        for p in image_paths:
            shutil.copy2(p, img_out / p.name)
            lp = lbl_in / (p.stem + ".txt")
            if lp.exists():
                shutil.copy2(lp, lbl_out / lp.name)
            orig_copied += 1

    for i, img_path in enumerate(image_paths):
        img = imread_unicode(img_path)
        if img is None:
            continue

        if randomize:
            c = int(np.random.randint(center_range[0], center_range[1] + 1))
            w = int(np.random.randint(width_range[0], width_range[1] + 1))
        else:
            c, w = center, width

        filtered = apply_filter(img, c, w, use_clahe)

        out_stem = f"{img_path.stem}_filt"
        imwrite_unicode(img_out / f"{out_stem}.jpg", filtered, output_jpeg_quality)

        lbl_path = lbl_in / (img_path.stem + ".txt")
        if lbl_path.exists():
            shutil.copy2(lbl_path, lbl_out / f"{out_stem}.txt")

        filtered_saved += 1
        if progress_callback:
            progress_callback(int(i / total * 100))

    for fname in ("data.yaml", "classes.txt"):
        src = ds / fname
        if src.exists():
            shutil.copy2(src, out / fname)

    if progress_callback:
        progress_callback(100)

    return {
        "originals": orig_copied,
        "filtered": filtered_saved,
        "total": orig_copied + filtered_saved,
        "output_dir": str(out),
    }
