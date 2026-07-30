from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Literal

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class AnalysisRequest:
    input_dir: Path
    output_dir: Path
    output_format: Literal["csv", "excel", "json"] = "csv"
    group_by_sample: bool = True
    save_visualizations: bool = True
    calibration_mode: Literal["relative", "calibrated"] = "relative"
    profile_path: Path | None = None
    segmentation_method: str = "auto"
    device: str = "cpu"
    exclude_white_tissue: bool = True
    id_pattern: str | None = None


@dataclass(frozen=True)
class ProgressEvent:
    current: int
    total: int
    image_path: Path | None
    status: str
    message: str
    successful: int = 0
    failed: int = 0


@dataclass(frozen=True)
class PreviewItem:
    image_path: Path
    visualization: np.ndarray
    features: Dict[str, float]
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class AnalysisResult:
    dataframe: pd.DataFrame
    table_path: Path
    raw_table_path: Path | None
    manifest_path: Path | None
    failure_path: Path | None
    visualization_dir: Path | None
    failures: tuple[Dict[str, str], ...]
    cancelled: bool


@dataclass(frozen=True)
class CalibrationImageRequest:
    training_image: Path
    validation_image: Path
    output_path: Path
    profile_id: str
    camera_id: str
    reference_id: str = "after_nov_2014"
    input_domain: str = "linear_srgb"
    white_balance: str = "none"
    exposure_normalization: str = "fixed_capture"
    raw_use_camera_wb: bool = True
    training_corners: np.ndarray | None = field(default=None, repr=False)
    validation_corners: np.ndarray | None = field(default=None, repr=False)


@dataclass(frozen=True)
class CalibrationImageResult:
    profile_path: Path
    profile_id: str
    status: str
    selected_model: str
    quality: Dict[str, Any]
    training_preview: np.ndarray
    validation_preview: np.ndarray
    training_corners: np.ndarray
    validation_corners: np.ndarray
    training_patch_csv: Path
    validation_patch_csv: Path
    warnings: tuple[str, ...] = ()


ProgressCallback = Callable[[ProgressEvent], None]

