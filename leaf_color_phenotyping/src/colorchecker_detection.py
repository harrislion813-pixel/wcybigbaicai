"""Detect and sample a classic 24-patch ColorChecker from an image."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

from .color_calibration import linear_to_srgb, srgb_to_linear
from .preprocessing import ImagePreprocessor
from .utils import COLORCHECKER_24_PATCH_IDS, RAW_IMAGE_EXTENSIONS, read_image_rgb


@dataclass(frozen=True)
class ColorCheckerExtraction:
    patch_ids: tuple[str, ...]
    rgb: np.ndarray
    corners: np.ndarray
    preview_rgb: np.ndarray
    warped_rgb: np.ndarray
    glare_fraction: float


def full_image_corners(image: np.ndarray) -> np.ndarray:
    height, width = image.shape[:2]
    return np.asarray(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
        dtype=np.float32,
    )


def _order_quad(points: np.ndarray) -> np.ndarray:
    quad = np.asarray(points, dtype=np.float32).reshape(4, 2)
    center = quad.mean(axis=0)
    angles = np.arctan2(quad[:, 1] - center[1], quad[:, 0] - center[0])
    cyclic = quad[np.argsort(angles)]
    start = int(np.argmin(cyclic[:, 0] + cyclic[:, 1]))
    cyclic = np.roll(cyclic, -start, axis=0)
    # The angle sort can run in either direction depending on the coordinate
    # convention. Force TL, TR, BR, BL.
    if cyclic[1, 0] < cyclic[-1, 0]:
        cyclic = cyclic[[0, 3, 2, 1]]
    return cyclic.astype(np.float32)


def detect_colorchecker_corners(display_rgb: np.ndarray) -> np.ndarray:
    """Find the largest plausible ColorChecker-like outer quadrilateral."""
    image = np.asarray(display_rgb)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("ColorChecker image must be an RGB image")
    if image.dtype != np.uint8:
        image = (np.clip(image, 0, 1) * 255).astype(np.uint8)
    height, width = image.shape[:2]
    scale = min(1.0, 1600.0 / max(height, width))
    small = cv2.resize(
        image, None, fx=scale, fy=scale,
        interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR,
    )
    # OpenCV's contrib MCC detector recognizes the internal 6x4 patch layout
    # and is substantially more reliable than an outer-contour heuristic when
    # the chart frame blends into the background.
    if hasattr(cv2, "mcc"):
        try:
            detector = cv2.mcc.CCheckerDetector_create()
            detector.setColorChartType(cv2.mcc.MCC24)
            detected = detector.process(cv2.cvtColor(small, cv2.COLOR_RGB2BGR), 1)
            if detected:
                checker = detector.getBestColorChecker()
                if checker is not None:
                    box = np.asarray(checker.getBox(), dtype=np.float32).reshape(-1, 2)
                    if box.shape == (4, 2):
                        return _order_quad(box / scale)
        except cv2.error:
            pass

    gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, 40, 130)
    edges = cv2.morphologyEx(
        edges, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8), iterations=2
    )
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    image_area = float(small.shape[0] * small.shape[1])
    candidates: list[tuple[float, np.ndarray]] = []
    for contour in contours:
        area = float(cv2.contourArea(contour))
        area_fraction = area / image_area
        if not 0.04 <= area_fraction <= 0.97:
            continue
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.025 * perimeter, True)
        if len(approx) != 4 or not cv2.isContourConvex(approx):
            continue
        quad = _order_quad(approx[:, 0, :])
        sides = np.linalg.norm(np.roll(quad, -1, axis=0) - quad, axis=1)
        short = max(1.0, float(min(np.mean(sides[[0, 2]]), np.mean(sides[[1, 3]]))))
        long = float(max(np.mean(sides[[0, 2]]), np.mean(sides[[1, 3]])))
        aspect = long / short
        if not 1.15 <= aspect <= 1.9:
            continue
        rectangularity = area / max(1.0, float(cv2.contourArea(cv2.convexHull(quad))))
        score = area_fraction * max(0.1, rectangularity)
        candidates.append((score, quad / scale))
    if not candidates:
        raise ValueError(
            "未能自动识别色卡外框；请使用手工四角，依次点击左上、右上、右下、左下"
        )
    return max(candidates, key=lambda item: item[0])[1].astype(np.float32)


def _warp(image: np.ndarray, corners: np.ndarray, width: int, height: int) -> np.ndarray:
    destination = np.asarray(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(corners.astype(np.float32), destination)
    return cv2.warpPerspective(image, matrix, (width, height), flags=cv2.INTER_LINEAR)


def _patch_regions(
    width: int, height: int, sample_fraction: float = 0.46
) -> Iterable[tuple[int, int, int, int]]:
    cell_width = width / 6.0
    cell_height = height / 4.0
    half_width = cell_width * sample_fraction / 2.0
    half_height = cell_height * sample_fraction / 2.0
    for row in range(4):
        for column in range(6):
            center_x = (column + 0.5) * cell_width
            center_y = (row + 0.5) * cell_height
            yield (
                max(0, int(round(center_x - half_width))),
                max(0, int(round(center_y - half_height))),
                min(width, int(round(center_x + half_width))),
                min(height, int(round(center_y + half_height))),
            )


def _sample_warped(warped: np.ndarray) -> tuple[np.ndarray, float]:
    patches = []
    glare_pixels = 0
    sampled_pixels = 0
    for x1, y1, x2, y2 in _patch_regions(warped.shape[1], warped.shape[0]):
        region = warped[y1:y2, x1:x2]
        flat = region.reshape(-1, 3)
        if len(flat) == 0:
            raise ValueError("色卡采样区域为空，请重新确认四角")
        patches.append(np.median(flat, axis=0))
        glare_pixels += int(np.count_nonzero(np.max(flat, axis=1) >= 0.995))
        sampled_pixels += len(flat)
    return np.asarray(patches, dtype=np.float64), glare_pixels / max(1, sampled_pixels)


def _orientation_score(samples: np.ndarray) -> float:
    grid = samples.reshape(4, 6, 3)
    neutral = grid[-1]
    neutral_chroma = np.mean(np.max(neutral, axis=1) - np.min(neutral, axis=1))
    colored_chroma = np.mean(
        np.max(grid[:3], axis=2) - np.min(grid[:3], axis=2)
    )
    luminance = neutral @ np.asarray([0.2126, 0.7152, 0.0722])
    ascending_error = float(np.sum(np.maximum(np.diff(luminance), 0)))
    separation_penalty = max(0.0, neutral_chroma - colored_chroma * 0.65)
    return float(neutral_chroma + 2.5 * ascending_error + separation_penalty)


def extract_colorchecker_patches(
    working_rgb: np.ndarray,
    *,
    display_rgb: np.ndarray | None = None,
    corners: np.ndarray | None = None,
    warp_size: tuple[int, int] = (900, 600),
) -> ColorCheckerExtraction:
    """Rectify a chart, choose its orientation, and sample its 24 patches.

    ``working_rgb`` must already be in the same normalized linear domain that
    will be used by the calibration profile. ``display_rgb`` is only used for
    contour detection and the user-facing preview.
    """
    working = np.asarray(working_rgb, dtype=np.float32)
    if working.ndim != 3 or working.shape[2] != 3:
        raise ValueError("working_rgb must have shape H x W x 3")
    if not np.isfinite(working).all() or np.any(working < 0) or np.any(working > 1):
        raise ValueError("working_rgb must be finite and normalized to [0, 1]")
    if display_rgb is None:
        display = linear_to_srgb(working)
    else:
        display = np.asarray(display_rgb, dtype=np.float32)
    if display.shape != working.shape:
        raise ValueError("display_rgb and working_rgb must have the same shape")
    base_corners = (
        detect_colorchecker_corners(display)
        if corners is None
        else _order_quad(np.asarray(corners, dtype=np.float32))
    )
    width, height = warp_size
    candidates = []
    for shift in range(4):
        oriented = np.roll(base_corners, -shift, axis=0)
        warped_working = _warp(working, oriented, width, height)
        samples, glare_fraction = _sample_warped(warped_working)
        candidates.append((
            _orientation_score(samples), oriented, warped_working, samples, glare_fraction
        ))
    _, selected_corners, warped_working, samples, glare_fraction = min(
        candidates, key=lambda item: item[0]
    )
    warped_display = _warp(display, selected_corners, width, height)
    preview = (np.clip(warped_display, 0, 1) * 255).astype(np.uint8)
    for index, (x1, y1, x2, y2) in enumerate(_patch_regions(width, height), start=1):
        cv2.rectangle(preview, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            preview, str(index), (x1 + 3, y1 + 18), cv2.FONT_HERSHEY_SIMPLEX,
            0.48, (255, 255, 255), 1, cv2.LINE_AA,
        )
    return ColorCheckerExtraction(
        patch_ids=tuple(COLORCHECKER_24_PATCH_IDS),
        rgb=samples,
        corners=selected_corners.copy(),
        preview_rgb=preview,
        warped_rgb=warped_working,
        glare_fraction=float(glare_fraction),
    )


def load_calibration_image(
    path: str | Path,
    *,
    input_domain: str = "linear_srgb",
    raw_use_camera_wb: bool = True,
) -> tuple[np.ndarray, np.ndarray, str]:
    """Load one calibration capture into working and display RGB arrays."""
    image_path = Path(path)
    if not image_path.is_file():
        raise FileNotFoundError(f"色卡图不存在: {image_path}")
    is_raw = image_path.suffix.lower() in RAW_IMAGE_EXTENSIONS
    input_kind = "raw" if is_raw else "rendered_rgb"
    if input_domain == "camera_linear_rgb" and not is_raw:
        raise ValueError("camera_linear_rgb 只支持 RAW 色卡图")
    if is_raw:
        working = ImagePreprocessor.read_raw(
            str(image_path), use_camera_wb=raw_use_camera_wb,
            output_bps=16, linear_output=True,
            output_color="raw" if input_domain == "camera_linear_rgb" else "srgb",
        )
        display = linear_to_srgb(working)
    else:
        display = read_image_rgb(image_path, as_float=True)
        if input_domain != "linear_srgb":
            raise ValueError("普通图片的 CCM 工作域必须是 linear_srgb")
        working = srgb_to_linear(display).astype(np.float32)
    return working.astype(np.float32), display.astype(np.float32), input_kind
