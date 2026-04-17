"""Image resolution consistency module.

Provides utilities to detect the target resolution of a dataset, check
whether images match it, resize mismatching images to the target size
(with letterbox/center-crop/stretch modes), and transform XLABEL
annotation coordinates to match the resized image.
"""

import copy
import os
import os.path as osp
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Dict, List, Optional, Tuple

from PIL import Image, UnidentifiedImageError

from anylabeling.views.labeling.logger import logger


class ResizeMode(str, Enum):
    """Supported resize strategies."""

    LETTERBOX = "letterbox"
    CENTER_CROP = "center_crop"
    STRETCH = "stretch"
    NONE = "none"


@dataclass
class ResizeResult:
    """Result of a single image resize operation."""

    success: bool
    output_path: str
    original_size: Tuple[int, int]
    target_size: Tuple[int, int]
    scale_x: float
    scale_y: float
    offset_x: int
    offset_y: int
    mode: ResizeMode
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Basic image inspection
# ---------------------------------------------------------------------------


def get_image_size(image_path: str) -> Optional[Tuple[int, int]]:
    """Return (width, height) of an image without loading pixel data.

    Returns ``None`` when the file cannot be read or is not an image.
    """
    try:
        with Image.open(image_path) as img:
            return img.size
    except (OSError, UnidentifiedImageError, ValueError) as exc:
        logger.debug("get_image_size failed for '%s': %s", image_path, exc)
        return None


# ---------------------------------------------------------------------------
# Core resize
# ---------------------------------------------------------------------------


def _safe_pil_mode(img: Image.Image, dst_path: str) -> Image.Image:
    """Convert image mode so it can be saved safely at dst_path.

    JPEG does not support alpha; convert to RGB for .jpg/.jpeg output.
    """
    ext = osp.splitext(dst_path)[1].lower()
    if ext in {".jpg", ".jpeg"} and img.mode != "RGB":
        return img.convert("RGB")
    return img


def resize_image(
    image_path: str,
    output_path: str,
    target_size: Tuple[int, int],
    mode: ResizeMode,
    pad_color: Tuple[int, int, int] = (0, 0, 0),
) -> ResizeResult:
    """Resize an image to target_size using the given mode.

    Args:
        image_path: Path to source image.
        output_path: Path to write resized image to.
        target_size: (target_width, target_height).
        mode: Resize strategy.
        pad_color: RGB tuple used for LETTERBOX padding.

    Returns:
        ResizeResult with transform metadata that can be used to
        update annotation coordinates.

    Safety: never overwrites the source unless ``output_path`` resolves
    to the same absolute path as ``image_path``.
    """
    target_w, target_h = int(target_size[0]), int(target_size[1])
    empty_result = ResizeResult(
        success=False,
        output_path=output_path,
        original_size=(0, 0),
        target_size=(target_w, target_h),
        scale_x=1.0,
        scale_y=1.0,
        offset_x=0,
        offset_y=0,
        mode=mode,
        error=None,
    )

    if target_w <= 0 or target_h <= 0:
        empty_result.error = f"Invalid target size: {target_size}"
        return empty_result

    try:
        with Image.open(image_path) as src:
            src.load()  # surface truncation errors early
            orig_w, orig_h = src.size
            empty_result.original_size = (orig_w, orig_h)

            if mode == ResizeMode.NONE or (
                orig_w == target_w and orig_h == target_h
            ):
                # No-op: still write if output path differs from input
                out = _safe_pil_mode(src.copy(), output_path)
                _ensure_parent(output_path)
                if osp.abspath(output_path) != osp.abspath(image_path):
                    out.save(output_path)
                return ResizeResult(
                    success=True,
                    output_path=output_path,
                    original_size=(orig_w, orig_h),
                    target_size=(target_w, target_h),
                    scale_x=1.0,
                    scale_y=1.0,
                    offset_x=0,
                    offset_y=0,
                    mode=ResizeMode.NONE,
                )

            if mode == ResizeMode.STRETCH:
                scaled = src.resize(
                    (target_w, target_h), Image.Resampling.LANCZOS
                )
                out = _safe_pil_mode(scaled, output_path)
                _ensure_parent(output_path)
                out.save(output_path)
                return ResizeResult(
                    success=True,
                    output_path=output_path,
                    original_size=(orig_w, orig_h),
                    target_size=(target_w, target_h),
                    scale_x=target_w / orig_w,
                    scale_y=target_h / orig_h,
                    offset_x=0,
                    offset_y=0,
                    mode=ResizeMode.STRETCH,
                )

            if mode == ResizeMode.LETTERBOX:
                scale = min(target_w / orig_w, target_h / orig_h)
                new_w = int(round(orig_w * scale))
                new_h = int(round(orig_h * scale))
                scaled = src.resize(
                    (new_w, new_h), Image.Resampling.LANCZOS
                )
                canvas = Image.new(
                    "RGB", (target_w, target_h), tuple(pad_color)
                )
                offset_x = (target_w - new_w) // 2
                offset_y = (target_h - new_h) // 2
                # Ensure source has compatible mode for paste
                if scaled.mode not in ("RGB", "RGBA"):
                    scaled = scaled.convert("RGB")
                canvas.paste(scaled, (offset_x, offset_y))
                out = _safe_pil_mode(canvas, output_path)
                _ensure_parent(output_path)
                out.save(output_path)
                return ResizeResult(
                    success=True,
                    output_path=output_path,
                    original_size=(orig_w, orig_h),
                    target_size=(target_w, target_h),
                    scale_x=scale,
                    scale_y=scale,
                    offset_x=offset_x,
                    offset_y=offset_y,
                    mode=ResizeMode.LETTERBOX,
                )

            if mode == ResizeMode.CENTER_CROP:
                scale = max(target_w / orig_w, target_h / orig_h)
                new_w = int(round(orig_w * scale))
                new_h = int(round(orig_h * scale))
                scaled = src.resize(
                    (new_w, new_h), Image.Resampling.LANCZOS
                )
                crop_x = (new_w - target_w) // 2
                crop_y = (new_h - target_h) // 2
                cropped = scaled.crop(
                    (crop_x, crop_y, crop_x + target_w, crop_y + target_h)
                )
                out = _safe_pil_mode(cropped, output_path)
                _ensure_parent(output_path)
                out.save(output_path)
                return ResizeResult(
                    success=True,
                    output_path=output_path,
                    original_size=(orig_w, orig_h),
                    target_size=(target_w, target_h),
                    scale_x=scale,
                    scale_y=scale,
                    offset_x=-crop_x,
                    offset_y=-crop_y,
                    mode=ResizeMode.CENTER_CROP,
                )

            empty_result.error = f"Unsupported mode: {mode}"
            return empty_result

    except FileNotFoundError as exc:
        empty_result.error = f"File not found: {exc}"
    except PermissionError as exc:
        empty_result.error = f"Permission denied: {exc}"
    except UnidentifiedImageError as exc:
        empty_result.error = f"Cannot identify image: {exc}"
    except (OSError, ValueError) as exc:
        empty_result.error = f"Image error: {exc}"

    logger.warning("resize_image failed for '%s': %s", image_path, empty_result.error)
    return empty_result


def _ensure_parent(path: str) -> None:
    parent = osp.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


# ---------------------------------------------------------------------------
# Annotation coordinate transformation
# ---------------------------------------------------------------------------


def transform_annotation(
    annotation_json: Dict, resize_result: ResizeResult
) -> Dict:
    """Return a new XLABEL annotation dict with coordinates transformed
    to match the resized image.

    Math:
        new_x = x * scale_x + offset_x
        new_y = y * scale_y + offset_y
    """
    result = copy.deepcopy(annotation_json)
    if not resize_result.success or resize_result.mode == ResizeMode.NONE:
        return result

    sx = resize_result.scale_x
    sy = resize_result.scale_y
    ox = resize_result.offset_x
    oy = resize_result.offset_y

    for shape in result.get("shapes", []):
        pts = shape.get("points", [])
        new_pts = []
        for p in pts:
            if (
                not isinstance(p, (list, tuple))
                or len(p) < 2
                or not all(isinstance(c, (int, float)) for c in p[:2])
            ):
                new_pts.append(p)
                continue
            new_x = p[0] * sx + ox
            new_y = p[1] * sy + oy
            new_pts.append([new_x, new_y])
        shape["points"] = new_pts

    result["imageWidth"] = resize_result.target_size[0]
    result["imageHeight"] = resize_result.target_size[1]
    return result


# ---------------------------------------------------------------------------
# Target resolution detection
# ---------------------------------------------------------------------------


def detect_target_resolution(
    image_paths: List[str],
    strategy: str = "mode",
    sample_size: int = 20,
) -> Tuple[int, int]:
    """Detect a good target resolution from a sample of images.

    Strategies: ``mode`` (most common), ``max``, ``min``, ``median``.
    Returns ``(0, 0)`` if no image can be read.
    """
    sample = image_paths[:sample_size] if sample_size else image_paths
    sizes: List[Tuple[int, int]] = []
    for p in sample:
        s = get_image_size(p)
        if s is not None:
            sizes.append(s)

    if not sizes:
        return (0, 0)

    if strategy == "mode":
        counter = Counter(sizes)
        return counter.most_common(1)[0][0]

    widths = [s[0] for s in sizes]
    heights = [s[1] for s in sizes]

    if strategy == "max":
        return (max(widths), max(heights))
    if strategy == "min":
        return (min(widths), min(heights))
    if strategy == "median":
        widths.sort()
        heights.sort()
        mid = len(widths) // 2
        return (widths[mid], heights[mid])

    logger.warning("Unknown strategy '%s', falling back to 'mode'", strategy)
    counter = Counter(sizes)
    return counter.most_common(1)[0][0]


def check_resolution_consistency(
    image_paths: List[str],
    target_size: Tuple[int, int],
    tolerance: int = 0,
) -> Dict:
    """Classify images as matching or mismatching the target size.

    Returns a dict with ``matching`` (list of paths) and
    ``mismatching`` (list of ``{"path", "size"}`` entries).
    """
    matching: List[str] = []
    mismatching: List[Dict] = []
    tw, th = target_size

    for p in image_paths:
        s = get_image_size(p)
        if s is None:
            mismatching.append({"path": p, "size": None, "error": "unreadable"})
            continue
        w, h = s
        if abs(w - tw) <= tolerance and abs(h - th) <= tolerance:
            matching.append(p)
        else:
            mismatching.append({"path": p, "size": (w, h)})

    return {"matching": matching, "mismatching": mismatching}


# ---------------------------------------------------------------------------
# Batch resize
# ---------------------------------------------------------------------------


def batch_resize(
    image_paths: List[str],
    output_dir: str,
    target_size: Tuple[int, int],
    mode: ResizeMode,
    pad_color: Tuple[int, int, int] = (0, 0, 0),
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> List[ResizeResult]:
    """Resize many images into ``output_dir`` (preserves base filenames).

    ``progress_callback(current_index, total)`` is called 1-indexed after
    each image, whether or not it succeeded.
    """
    results: List[ResizeResult] = []
    total = len(image_paths)
    os.makedirs(output_dir, exist_ok=True)

    for i, src in enumerate(image_paths, start=1):
        dst = osp.join(output_dir, osp.basename(src))
        try:
            res = resize_image(src, dst, target_size, mode, pad_color)
        except Exception as exc:
            logger.error("Unexpected error resizing '%s': %s", src, exc)
            res = ResizeResult(
                success=False,
                output_path=dst,
                original_size=(0, 0),
                target_size=target_size,
                scale_x=1.0,
                scale_y=1.0,
                offset_x=0,
                offset_y=0,
                mode=mode,
                error=str(exc),
            )
        results.append(res)
        if progress_callback is not None:
            try:
                progress_callback(i, total)
            except Exception:
                pass

    return results
