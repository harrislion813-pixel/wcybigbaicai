"""Application-facing workflows for fitting and validating CCM profiles.

The command-line script and desktop UI both need the same scientific gates.
This module keeps those rules independent from either presentation layer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
from pathlib import Path
from typing import Any, Dict, Mapping

import numpy as np

from .color_calibration import (
    CalibrationModelFit,
    fit_rgb_to_xyz_model,
    lab_d50_to_xyz_d65,
    load_calibration_profile,
    rgb_to_xyz_delta_e00,
    validate_rgb_to_xyz_model,
    write_calibration_profile,
    xyz_d65_to_srgb,
)
from .utils import COLORCHECKER_24_PATCH_IDS, get_colorchecker_reference_lab


VEGETATION_PATCH_IDS = {
    "foliage", "bluish_green", "yellow_green", "green", "cyan",
}
NEUTRAL_PATCH_IDS = {
    "white_95", "neutral_8", "neutral_65", "neutral_5", "neutral_35", "black_2",
}


@dataclass(frozen=True)
class CalibrationGates:
    median_max: float = 2.5
    p95_max: float = 6.0
    vegetation_mean_max: float = 3.0
    neutral_mean_max: float = 2.0

    def as_dict(self) -> Dict[str, float]:
        values = {
            "median_max": float(self.median_max),
            "p95_max": float(self.p95_max),
            "vegetation_mean_max": float(self.vegetation_mean_max),
            "neutral_mean_max": float(self.neutral_mean_max),
        }
        array = np.asarray(list(values.values()), dtype=np.float64)
        if not np.isfinite(array).all() or np.any(array < 0):
            raise ValueError("Delta E quality gates must be finite and non-negative")
        return values


@dataclass(frozen=True)
class CalibrationProfileRequest:
    training_rgb: np.ndarray
    validation_rgb: np.ndarray
    training_source: Path
    validation_source: Path
    output_path: Path
    profile_id: str
    camera_id: str
    input_kind: str
    reference_id: str
    input_domain: str = "linear_srgb"
    white_balance: str = "none"
    exposure_normalization: str = "fixed_capture"
    raw_use_camera_wb: bool = True
    gray_card_rgb: tuple[float, float, float] | None = None
    model: str = "auto"
    source_scale: str = "1"
    gates: CalibrationGates = field(default_factory=CalibrationGates)


@dataclass(frozen=True)
class CalibrationProfileResult:
    profile_path: Path
    profile_id: str
    status: str
    selected_model: str
    quality: Dict[str, Any]


def sha256_file(path: str | Path) -> str:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Calibration source does not exist: {source}")
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def delta_e_metrics(values: np.ndarray) -> Dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or len(array) == 0 or not np.isfinite(array).all():
        raise ValueError("Delta E values must be a non-empty finite vector")
    return {
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p95": float(np.percentile(array, 95)),
        "max": float(np.max(array)),
    }


def choose_calibration_model(
    fits: Mapping[str, CalibrationModelFit],
    validation_metrics: Mapping[str, Mapping[str, float]],
    *,
    root_min_median_improvement: float = 0.10,
) -> str:
    """Use root-polynomial only for a material, non-regressive gain."""
    if set(fits) == {"linear_3x3"}:
        return "linear_3x3"
    if set(fits) == {"root_polynomial_2"}:
        return "root_polynomial_2"
    linear = validation_metrics["linear_3x3"]
    root = validation_metrics["root_polynomial_2"]
    median_limit = linear["median"] * (1.0 - root_min_median_improvement)
    if root["median"] <= median_limit and root["p95"] <= linear["p95"]:
        return "root_polynomial_2"
    return "linear_3x3"


def _validate_samples(samples: np.ndarray, label: str) -> np.ndarray:
    rgb = np.asarray(samples, dtype=np.float64)
    expected = (len(COLORCHECKER_24_PATCH_IDS), 3)
    if rgb.shape != expected:
        raise ValueError(f"{label} patch RGB must have shape {expected}, got {rgb.shape}")
    if not np.isfinite(rgb).all() or np.any(rgb < 0) or np.any(rgb > 1):
        raise ValueError(f"{label} patch RGB must be finite and normalized to [0, 1]")
    return rgb


def _validate_independent_patch_samples(
    training_rgb: np.ndarray,
    validation_rgb: np.ndarray,
) -> Dict[str, float]:
    """Reject duplicated validation samples and report their separation.

    Different source files are necessary but not sufficient evidence of an
    independent capture: an image can be re-encoded, renamed, or changed only
    outside the sampled chart.  Comparing the actual 24-patch measurements
    closes that loophole while retaining a quantitative audit value.
    """
    absolute_difference = np.abs(training_rgb - validation_rgb)
    metrics = {
        "mean_absolute_rgb_difference": float(np.mean(absolute_difference)),
        "max_absolute_rgb_difference": float(np.max(absolute_difference)),
    }
    if np.allclose(training_rgb, validation_rgb, rtol=0.0, atol=1e-6):
        raise ValueError(
            "Training and validation patch measurements are identical or "
            "nearly identical; use a genuinely independent capture"
        )
    return metrics


def _fit_candidates(
    training_rgb: np.ndarray,
    validation_rgb: np.ndarray,
    target_xyz: np.ndarray,
    requested_model: str,
) -> tuple[CalibrationModelFit, Dict[str, Dict[str, float]], str]:
    if requested_model not in {"auto", "linear_3x3", "root_polynomial_2"}:
        raise ValueError("model must be auto, linear_3x3, or root_polynomial_2")
    model_types = (
        ("linear_3x3", "root_polynomial_2")
        if requested_model == "auto"
        else (requested_model,)
    )
    fits: Dict[str, CalibrationModelFit] = {}
    validation: Dict[str, Dict[str, float]] = {}
    for model_type in model_types:
        fit = fit_rgb_to_xyz_model(training_rgb, target_xyz, model_type=model_type)
        fits[model_type] = fit
        validation[model_type] = validate_rgb_to_xyz_model(
            validation_rgb, target_xyz, fit
        )
    selected = choose_calibration_model(fits, validation)
    return fits[selected], validation, selected


def _quality_report(
    fit: CalibrationModelFit,
    validation_rgb: np.ndarray,
    target_xyz: np.ndarray,
    gates: CalibrationGates,
) -> Dict[str, Any]:
    patch_ids = tuple(COLORCHECKER_24_PATCH_IDS)
    delta_e = rgb_to_xyz_delta_e00(validation_rgb, target_xyz, fit)
    id_to_index = {patch_id: index for index, patch_id in enumerate(patch_ids)}
    vegetation = delta_e[[id_to_index[item] for item in VEGETATION_PATCH_IDS]]
    neutral = delta_e[[id_to_index[item] for item in NEUTRAL_PATCH_IDS]]
    validation = delta_e_metrics(delta_e)
    vegetation_metrics = delta_e_metrics(vegetation)
    neutral_metrics = delta_e_metrics(neutral)
    gate_values = gates.as_dict()
    failures = []
    if validation["median"] > gate_values["median_max"]:
        failures.append("validation median Delta E 00")
    if validation["p95"] > gate_values["p95_max"]:
        failures.append("validation p95 Delta E 00")
    if vegetation_metrics["mean"] > gate_values["vegetation_mean_max"]:
        failures.append("vegetation-patch mean Delta E 00")
    if neutral_metrics["mean"] > gate_values["neutral_mean_max"]:
        failures.append("neutral-patch mean Delta E 00")
    return {
        "passed": not failures,
        "failures": failures,
        "rank": fit.rank,
        "condition_number": fit.condition_number,
        "training_delta_e00": fit.training_delta_e00,
        "validation_delta_e00": validation,
        "vegetation_validation_delta_e00": vegetation_metrics,
        "neutral_validation_delta_e00": neutral_metrics,
        "gates": gate_values,
        "validation_patch_delta_e00": {
            patch_id: float(delta_e[index]) for index, patch_id in enumerate(patch_ids)
        },
        "outlier_patch_ids": [
            patch_id for index, patch_id in enumerate(patch_ids)
            if delta_e[index] > gate_values["p95_max"]
        ],
    }


def reference_metadata(reference_id: str) -> Dict[str, str]:
    if reference_id not in {"before_nov_2014", "after_nov_2014"}:
        raise ValueError("Unsupported ColorChecker reference version")
    source = (
        "BabelColor/X-Rite pre-November-2014 data"
        if reference_id == "before_nov_2014"
        else "X-Rite post-November-2014 data"
    )
    return {
        "chart": "ColorChecker Classic 24",
        "id": reference_id,
        "source": source,
        "source_space": "CIE Lab D50",
    }


def fit_calibration_profile(request: CalibrationProfileRequest) -> CalibrationProfileResult:
    """Fit, independently validate, and persist a version-2 profile."""
    if not request.profile_id.strip() or not request.camera_id.strip():
        raise ValueError("profile_id and camera_id must not be empty")
    if request.input_kind not in {"raw", "rendered_rgb"}:
        raise ValueError("input_kind must be raw or rendered_rgb")
    if request.input_domain not in {"linear_srgb", "camera_linear_rgb"}:
        raise ValueError("image workflow profiles require a linear RGB input domain")
    if request.input_domain == "camera_linear_rgb" and request.input_kind != "raw":
        raise ValueError("camera_linear_rgb profiles require RAW images")
    if request.white_balance == "gray_card" and request.gray_card_rgb is None:
        raise ValueError("gray-card white balance requires gray_card_rgb")
    if request.white_balance != "gray_card" and request.gray_card_rgb is not None:
        raise ValueError("gray_card_rgb is only valid with gray-card white balance")

    training_rgb = _validate_samples(request.training_rgb, "training")
    validation_rgb = _validate_samples(request.validation_rgb, "validation")
    training_path = Path(request.training_source).resolve()
    validation_path = Path(request.validation_source).resolve()
    training_sha = sha256_file(training_path)
    validation_sha = sha256_file(validation_path)
    if training_path == validation_path or training_sha == validation_sha:
        raise ValueError("Training and validation captures must be independent")
    independence_metrics = _validate_independent_patch_samples(
        training_rgb, validation_rgb
    )

    target_xyz = lab_d50_to_xyz_d65(
        get_colorchecker_reference_lab(request.reference_id)
    )
    fit, candidates, selected_model = _fit_candidates(
        training_rgb, validation_rgb, target_xyz, request.model
    )
    quality = _quality_report(fit, validation_rgb, target_xyz, request.gates)
    quality["candidate_validation_delta_e00"] = candidates
    quality["capture_independence"] = independence_metrics
    reference_srgb, _ = xyz_d65_to_srgb(target_xyz)
    quality["reference_display_out_of_gamut_fraction"] = float(np.mean(
        np.any((reference_srgb < 0) | (reference_srgb > 1), axis=1)
    ))

    gray_card = list(request.gray_card_rgb) if request.gray_card_rgb is not None else None
    profile = {
        "schema_version": 2,
        "profile_id": request.profile_id.strip(),
        "status": "validated" if quality["passed"] else "draft",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "input": {
            "kind": request.input_kind,
            "domain": request.input_domain,
            "range": [0.0, 1.0],
            "source_scale": request.source_scale,
            "channel_order": "RGB",
            "camera_id": request.camera_id.strip(),
        },
        "preprocessing": {
            "white_balance": request.white_balance,
            "exposure_normalization": request.exposure_normalization,
            "raw_use_camera_wb": bool(request.raw_use_camera_wb),
            "gray_card_rgb": gray_card,
        },
        "model": {
            "type": selected_model,
            "degree": 2 if selected_model == "root_polynomial_2" else 1,
            "matrix_layout": "features_by_output",
            "matrix": fit.matrix.tolist(),
        },
        "target": {
            "space": "XYZ",
            "illuminant": "D65",
            "observer": "2_degree",
            "display_encoding": "sRGB with explicit gamut clipping only for display",
        },
        "reference": reference_metadata(request.reference_id),
        "datasets": {
            "training": {
                "path": str(training_path),
                "sha256": training_sha,
                "patch_count": len(COLORCHECKER_24_PATCH_IDS),
            },
            "validation": {
                "path": str(validation_path),
                "sha256": validation_sha,
                "patch_count": len(COLORCHECKER_24_PATCH_IDS),
                "independent": True,
                "content_distinct_from_training": True,
            },
        },
        "quality": quality,
        "integrity": {},
    }
    output = write_calibration_profile(profile, request.output_path)
    return CalibrationProfileResult(
        profile_path=output.resolve(),
        profile_id=request.profile_id.strip(),
        status=profile["status"],
        selected_model=selected_model,
        quality=quality,
    )


def inspect_calibration_profile(path: str | Path) -> Dict[str, Any]:
    profile = load_calibration_profile(path)
    return {
        "profile_id": profile.profile_id,
        "status": profile.status,
        "sha256": profile.sha256,
        "input": profile.data["input"],
        "preprocessing": profile.data["preprocessing"],
        "model": {
            "type": profile.model_type,
            "matrix_shape": list(profile.matrix.shape),
        },
        "reference": profile.data["reference"],
        "quality": profile.data["quality"],
    }
