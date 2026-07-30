"""Versioned, self-describing color-calibration profiles.

The batch runtime intentionally keeps profile application lightweight. Profile
creation and advanced colour-science fitting live in the optional calibration
tooling, while every runtime artifact remains independently auditable.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Mapping

import numpy as np

from .utils import BRADFORD_D50_TO_D65, delta_e_2000, rgb_to_xyz, xyz_to_lab


PROFILE_SCHEMA_VERSION = 2
SUPPORTED_INPUT_DOMAINS = {
    "encoded_srgb",
    "linear_srgb",
    "camera_linear_rgb",
}
SUPPORTED_MODEL_TYPES = {
    "linear_3x3",
    "legacy_polynomial",
    "root_polynomial_2",
}
SUPPORTED_REFERENCE_IDS = {
    "before_nov_2014",
    "after_nov_2014",
    "custom",
}
_VALIDATION_METRICS = {"mean", "median", "p95", "max"}


def _validate_delta_e_metrics(container: Mapping[str, Any], key: str) -> None:
    metrics = container.get(key)
    if not isinstance(metrics, dict) or not _VALIDATION_METRICS.issubset(metrics):
        raise ValueError(
            f"validated calibration profiles require {key} with "
            "mean, median, p95, and max"
        )
    try:
        values = [float(metrics[name]) for name in _VALIDATION_METRICS]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{key} contains a non-numeric value") from exc
    if not np.isfinite(values).all() or np.any(np.asarray(values) < 0):
        raise ValueError(f"{key} must contain finite non-negative values")


def _is_sha256(value: Any) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


@dataclass(frozen=True)
class CalibrationModelFit:
    """Numerical result of fitting an RGB-to-XYZ calibration model."""

    model_type: str
    matrix: np.ndarray
    rank: int
    condition_number: float
    residuals: np.ndarray
    training_delta_e00: Dict[str, float]


@dataclass(frozen=True)
class CalibrationApplication:
    """Unclipped colorimetric output plus display RGB and clipping diagnostics."""

    xyz_d65: np.ndarray
    lab_d65: np.ndarray
    srgb: np.ndarray
    qc: Dict[str, float]


def srgb_to_linear(srgb: np.ndarray) -> np.ndarray:
    """Decode IEC sRGB values without clipping."""
    values = np.asarray(srgb, dtype=np.float64)
    return np.where(
        values <= 0.04045,
        values / 12.92,
        np.power(np.maximum((values + 0.055) / 1.055, 0), 2.4),
    )


def linear_to_srgb(linear_rgb: np.ndarray) -> np.ndarray:
    """Encode linear sRGB values without gamut clipping."""
    values = np.asarray(linear_rgb, dtype=np.float64)
    return np.where(
        values <= 0.0031308,
        12.92 * values,
        1.055 * np.power(np.maximum(values, 0), 1 / 2.4) - 0.055,
    )


def lab_d50_to_xyz_d65(reference_lab: np.ndarray) -> np.ndarray:
    """Convert D50 Lab references to D65 XYZ without an sRGB gamut round-trip."""
    lab = np.asarray(reference_lab, dtype=np.float64)
    if lab.ndim != 2 or lab.shape[1:] != (3,):
        raise ValueError("reference_lab must have shape (N, 3)")
    if not np.isfinite(lab).all():
        raise ValueError("reference_lab contains NaN or infinite values")

    L, a, b = lab[:, 0], lab[:, 1], lab[:, 2]
    fy = (L + 16) / 116
    fx = a / 500 + fy
    fz = fy - b / 200
    delta = 6 / 29

    def inverse_curve(values: np.ndarray) -> np.ndarray:
        return np.where(
            values > delta,
            values ** 3,
            3 * delta ** 2 * (values - 4 / 29),
        )

    xyz_d50 = np.column_stack([
        inverse_curve(fx) * 0.96422,
        inverse_curve(fy),
        inverse_curve(fz) * 0.82521,
    ])
    return (BRADFORD_D50_TO_D65 @ xyz_d50.T).T


def xyz_d65_to_srgb(xyz_d65: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return unclipped and clipped display sRGB for D65 XYZ values."""
    xyz = np.asarray(xyz_d65, dtype=np.float64)
    matrix = np.array([
        [3.2404542, -1.5371385, -0.4985314],
        [-0.9692660, 1.8760108, 0.0415560],
        [0.0556434, -0.2040259, 1.0572252],
    ], dtype=np.float64)
    linear_rgb = xyz.reshape(-1, 3) @ matrix.T
    encoded = linear_to_srgb(linear_rgb).reshape(xyz.shape)
    return encoded, np.clip(encoded, 0, 1)


def root_polynomial_expansion_2(rgb: np.ndarray) -> np.ndarray:
    """Exposure-homogeneous degree-2 root-polynomial RGB basis."""
    values = np.asarray(rgb, dtype=np.float64)
    if values.ndim != 2 or values.shape[1:] != (3,):
        raise ValueError("root-polynomial input must have shape (N, 3)")
    if not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("root-polynomial input must contain finite non-negative RGB")
    R, G, B = values.T
    return np.column_stack([
        R,
        G,
        B,
        np.sqrt(R * G),
        np.sqrt(R * B),
        np.sqrt(G * B),
    ])


def _model_design_matrix(rgb: np.ndarray, model_type: str) -> np.ndarray:
    if model_type == "linear_3x3":
        return np.asarray(rgb, dtype=np.float64)
    if model_type == "root_polynomial_2":
        return root_polynomial_expansion_2(rgb)
    raise ValueError(
        "XYZ calibration fitting supports 'linear_3x3' and 'root_polynomial_2'"
    )


def _delta_e_metrics(delta_e: np.ndarray) -> Dict[str, float]:
    values = np.asarray(delta_e, dtype=np.float64)
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95)),
        "max": float(np.max(values)),
    }


def fit_rgb_to_xyz_model(
    measured_rgb: np.ndarray,
    target_xyz_d65: np.ndarray,
    *,
    model_type: str = "linear_3x3",
    max_condition_number: float = 1e8,
) -> CalibrationModelFit:
    """Fit a linear or root-polynomial RGB-to-XYZ model with hard diagnostics."""
    measured = np.asarray(measured_rgb, dtype=np.float64)
    target = np.asarray(target_xyz_d65, dtype=np.float64)
    if measured.ndim != 2 or measured.shape[1:] != (3,):
        raise ValueError("measured_rgb must have shape (N, 3)")
    if target.ndim != 2 or target.shape[1:] != (3,):
        raise ValueError("target_xyz_d65 must have shape (N, 3)")
    if len(measured) == 0 or len(measured) != len(target):
        raise ValueError("measured_rgb and target_xyz_d65 must be non-empty and paired")
    if not np.isfinite(measured).all() or not np.isfinite(target).all():
        raise ValueError("calibration samples contain NaN or infinite values")
    if np.any(measured < 0) or np.any(measured > 1):
        raise ValueError("measured_rgb must be normalized to [0, 1]")

    design = _model_design_matrix(measured, model_type)
    terms = design.shape[1]
    if len(measured) < 2 * terms:
        raise ValueError(
            f"{model_type} requires at least {2 * terms} patches; received {len(measured)}"
        )
    rank = int(np.linalg.matrix_rank(design))
    if rank != terms:
        raise ValueError(
            f"calibration design matrix is rank deficient: rank={rank}, terms={terms}"
        )
    condition_number = float(np.linalg.cond(design))
    if not np.isfinite(condition_number) or condition_number > max_condition_number:
        raise ValueError(
            f"calibration design matrix condition number {condition_number:.6g} "
            f"exceeds {max_condition_number:.6g}"
        )

    matrix, residuals, fitted_rank, _ = np.linalg.lstsq(design, target, rcond=None)
    if int(fitted_rank) != terms:
        raise ValueError("calibration least-squares fit lost rank")
    fitted_xyz = design @ matrix
    delta_e = delta_e_2000(xyz_to_lab(fitted_xyz), xyz_to_lab(target))
    return CalibrationModelFit(
        model_type=model_type,
        matrix=matrix,
        rank=rank,
        condition_number=condition_number,
        residuals=np.asarray(residuals, dtype=np.float64),
        training_delta_e00=_delta_e_metrics(delta_e),
    )


def validate_rgb_to_xyz_model(
    measured_rgb: np.ndarray,
    reference_xyz_d65: np.ndarray,
    fit: CalibrationModelFit,
) -> Dict[str, float]:
    """Evaluate an already fitted model on independent paired samples."""
    measured = np.asarray(measured_rgb, dtype=np.float64)
    reference = np.asarray(reference_xyz_d65, dtype=np.float64)
    if measured.shape != reference.shape or measured.ndim != 2 or measured.shape[1:] != (3,):
        raise ValueError("validation RGB and XYZ arrays must have matching shape (N, 3)")
    if len(measured) == 0 or not np.isfinite(measured).all() or not np.isfinite(reference).all():
        raise ValueError("validation arrays must be non-empty and finite")
    if np.any(measured < 0) or np.any(measured > 1):
        raise ValueError("validation RGB must be normalized to [0, 1]")
    delta_e = rgb_to_xyz_delta_e00(measured, reference, fit)
    return _delta_e_metrics(delta_e)


def predict_rgb_to_xyz_model(
    measured_rgb: np.ndarray,
    fit: CalibrationModelFit,
) -> np.ndarray:
    """Predict D65 XYZ values for normalized RGB samples."""
    measured = np.asarray(measured_rgb, dtype=np.float64)
    if measured.ndim != 2 or measured.shape[1:] != (3,):
        raise ValueError("measured_rgb must have shape (N, 3)")
    if len(measured) == 0 or not np.isfinite(measured).all():
        raise ValueError("measured_rgb must be non-empty and finite")
    if np.any(measured < 0) or np.any(measured > 1):
        raise ValueError("measured_rgb must be normalized to [0, 1]")
    return _model_design_matrix(measured, fit.model_type) @ fit.matrix


def rgb_to_xyz_delta_e00(
    measured_rgb: np.ndarray,
    reference_xyz_d65: np.ndarray,
    fit: CalibrationModelFit,
) -> np.ndarray:
    """Return per-patch Delta E 2000 errors for a fitted RGB-to-XYZ model."""
    reference = np.asarray(reference_xyz_d65, dtype=np.float64)
    predicted = predict_rgb_to_xyz_model(measured_rgb, fit)
    if reference.shape != predicted.shape or not np.isfinite(reference).all():
        raise ValueError("reference_xyz_d65 must be finite and match measured_rgb")
    return np.asarray(
        delta_e_2000(xyz_to_lab(predicted), xyz_to_lab(reference)),
        dtype=np.float64,
    )


def apply_calibration_profile(
    profile: "CalibrationProfile",
    rgb: np.ndarray,
    mask: np.ndarray | None = None,
) -> CalibrationApplication:
    """Apply a profile while preserving unclipped XYZ and reporting gamut loss."""
    if profile.status != "validated":
        raise ValueError("Only validated calibration profiles may be applied")
    image = np.asarray(rgb, dtype=np.float64)
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError("rgb must have shape (H, W, 3)")
    if not np.isfinite(image).all() or np.any(image < 0) or np.any(image > 1):
        raise ValueError("rgb must contain finite values normalized to [0, 1]")
    if mask is None:
        selected = np.ones(image.shape[:2], dtype=bool)
    else:
        selected = np.asarray(mask) != 0
        if selected.shape != image.shape[:2]:
            raise ValueError("mask must match the RGB image height and width")
        if not np.any(selected):
            raise ValueError("mask must select at least one pixel")

    flat = image.reshape(-1, 3)
    design = _model_design_matrix(flat, profile.model_type)
    transformed = (design @ profile.matrix).reshape(image.shape)
    if profile.data["target"]["space"] == "XYZ":
        xyz = transformed
        srgb_unclipped, srgb = xyz_d65_to_srgb(xyz)
    else:
        srgb_unclipped = transformed
        srgb = np.clip(srgb_unclipped, 0, 1)
        xyz = rgb_to_xyz(srgb.astype(np.float32)).astype(np.float64)

    below = srgb_unclipped < 0
    above = srgb_unclipped > 1
    clipped = below | above
    qc = {
        "QC_CCM_negative_fraction": float(np.mean(np.any(below, axis=-1)[selected])),
        "QC_CCM_clipped_fraction": float(np.mean(np.any(clipped, axis=-1)[selected])),
        "QC_CCM_R_clipped_fraction": float(np.mean(clipped[..., 0][selected])),
        "QC_CCM_G_clipped_fraction": float(np.mean(clipped[..., 1][selected])),
        "QC_CCM_B_clipped_fraction": float(np.mean(clipped[..., 2][selected])),
    }
    return CalibrationApplication(
        xyz_d65=xyz.astype(np.float64),
        lab_d65=xyz_to_lab(xyz).astype(np.float32),
        srgb=srgb.astype(np.float32),
        qc=qc,
    )


def _canonical_payload(profile: Mapping[str, Any]) -> bytes:
    payload = deepcopy(dict(profile))
    integrity = payload.get("integrity")
    if isinstance(integrity, dict):
        integrity.pop("sha256", None)
        if not integrity:
            payload.pop("integrity", None)
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def calibration_profile_sha256(profile: Mapping[str, Any]) -> str:
    """Hash profile content, excluding the self-referential SHA-256 field."""
    return hashlib.sha256(_canonical_payload(profile)).hexdigest()


def _require_mapping(profile: Mapping[str, Any], key: str) -> Dict[str, Any]:
    value = profile.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"calibration profile '{key}' must be an object")
    return value


def _expected_matrix_shape(model: Mapping[str, Any]) -> tuple[int, int]:
    model_type = model.get("type")
    if model_type == "linear_3x3":
        return (3, 3)
    if model_type == "root_polynomial_2":
        return (6, 3)
    if model_type == "legacy_polynomial":
        degree = model.get("degree")
        terms = {1: 3, 2: 9, 3: 19}.get(degree)
        if terms is None:
            raise ValueError("legacy_polynomial degree must be one of [1, 2, 3]")
        return (terms, 3)
    raise ValueError(
        f"Unsupported calibration profile model type '{model_type}'; "
        f"use one of {sorted(SUPPORTED_MODEL_TYPES)}"
    )


@dataclass(frozen=True)
class CalibrationProfile:
    """Validated representation of a version-2 CCM profile."""

    data: Dict[str, Any]

    @classmethod
    def from_mapping(
        cls,
        profile: Mapping[str, Any],
        *,
        verify_integrity: bool = True,
    ) -> "CalibrationProfile":
        data = deepcopy(dict(profile))
        if data.get("schema_version") != PROFILE_SCHEMA_VERSION:
            raise ValueError(
                f"calibration profile schema_version must be {PROFILE_SCHEMA_VERSION}"
            )
        profile_id = data.get("profile_id")
        if not isinstance(profile_id, str) or not profile_id.strip():
            raise ValueError("calibration profile profile_id must be a non-empty string")
        if data.get("status") not in {"draft", "validated"}:
            raise ValueError("calibration profile status must be 'draft' or 'validated'")
        created_at = data.get("created_at")
        if not isinstance(created_at, str) or not created_at.strip():
            raise ValueError("calibration profile created_at must be a non-empty string")

        input_cfg = _require_mapping(data, "input")
        if input_cfg.get("kind") not in {"raw", "rendered_rgb", "jpeg"}:
            raise ValueError(
                "calibration profile input.kind must be 'raw' or 'rendered_rgb'"
            )
        if input_cfg.get("domain") not in SUPPORTED_INPUT_DOMAINS:
            raise ValueError(
                "calibration profile input.domain must be one of "
                f"{sorted(SUPPORTED_INPUT_DOMAINS)}"
            )
        if input_cfg.get("range") != [0.0, 1.0] and input_cfg.get("range") != [0, 1]:
            raise ValueError("calibration profile input.range must be [0, 1]")
        if input_cfg.get("channel_order") != "RGB":
            raise ValueError("calibration profile input.channel_order must be 'RGB'")
        if (
            input_cfg.get("domain") == "camera_linear_rgb"
            and input_cfg.get("kind") != "raw"
        ):
            raise ValueError("camera_linear_rgb calibration profiles require raw input")
        camera_id = input_cfg.get("camera_id")
        if not isinstance(camera_id, str) or not camera_id.strip():
            raise ValueError("calibration profile input.camera_id must be a non-empty string")

        preprocessing = _require_mapping(data, "preprocessing")
        for key in ("white_balance", "exposure_normalization"):
            if not isinstance(preprocessing.get(key), str) or not preprocessing[key].strip():
                raise ValueError(
                    f"calibration profile preprocessing.{key} must be non-empty"
                )
        gray_card_rgb = preprocessing.get("gray_card_rgb")
        if preprocessing["white_balance"] == "gray_card":
            gray_card = np.asarray(gray_card_rgb, dtype=np.float64)
            if (
                gray_card.shape != (3,)
                or not np.isfinite(gray_card).all()
                or np.any(gray_card <= 0)
                or np.any(gray_card > 1)
            ):
                raise ValueError(
                    "gray_card profiles require finite preprocessing.gray_card_rgb "
                    "values in (0, 1]"
                )
            preprocessing["gray_card_rgb"] = gray_card.tolist()
        elif gray_card_rgb is not None:
            raise ValueError(
                "preprocessing.gray_card_rgb is only valid with white_balance=gray_card"
            )
        if not isinstance(preprocessing.get("raw_use_camera_wb"), bool):
            raise ValueError(
                "calibration profile preprocessing.raw_use_camera_wb must be boolean"
            )

        model = _require_mapping(data, "model")
        expected_shape = _expected_matrix_shape(model)
        matrix = np.asarray(model.get("matrix"), dtype=np.float64)
        if matrix.shape != expected_shape:
            raise ValueError(
                f"calibration profile matrix shape {matrix.shape} does not match "
                f"model {model.get('type')}; expected {expected_shape}"
            )
        if not np.isfinite(matrix).all():
            raise ValueError("calibration profile matrix contains NaN or infinite values")
        if model.get("matrix_layout", "features_by_output") != "features_by_output":
            raise ValueError(
                "calibration profile model.matrix_layout must be 'features_by_output'"
            )
        expected_degree = {
            "linear_3x3": 1,
            "root_polynomial_2": 2,
        }.get(model.get("type"))
        if expected_degree is not None and model.get("degree") != expected_degree:
            raise ValueError(
                f"calibration profile {model.get('type')} degree must be {expected_degree}"
            )

        target = _require_mapping(data, "target")
        if target.get("space") not in {"sRGB", "XYZ"}:
            raise ValueError("calibration profile target.space must be 'sRGB' or 'XYZ'")
        if target.get("illuminant") != "D65":
            raise ValueError("calibration profile target.illuminant must be 'D65'")
        if target.get("space") == "XYZ" and input_cfg.get("domain") == "encoded_srgb":
            raise ValueError(
                "XYZ calibration profiles require linear_srgb or camera_linear_rgb input"
            )
        if model.get("type") == "root_polynomial_2" and target.get("space") != "XYZ":
            raise ValueError("root_polynomial_2 profiles must target XYZ")

        reference = _require_mapping(data, "reference")
        if reference.get("id") not in SUPPORTED_REFERENCE_IDS:
            raise ValueError(
                "calibration profile reference.id must be one of "
                f"{sorted(SUPPORTED_REFERENCE_IDS)}"
            )

        quality = _require_mapping(data, "quality")
        if data["status"] == "validated":
            if quality.get("passed") is not True:
                raise ValueError("validated calibration profiles require quality.passed=true")
            for metric_key in (
                "training_delta_e00",
                "validation_delta_e00",
                "vegetation_validation_delta_e00",
                "neutral_validation_delta_e00",
            ):
                _validate_delta_e_metrics(quality, metric_key)
            gates = quality.get("gates")
            if not isinstance(gates, dict):
                raise ValueError("validated calibration profiles require quality.gates")
            gate_checks = (
                ("validation_delta_e00", "median", "median_max"),
                ("validation_delta_e00", "p95", "p95_max"),
                (
                    "vegetation_validation_delta_e00",
                    "mean",
                    "vegetation_mean_max",
                ),
                ("neutral_validation_delta_e00", "mean", "neutral_mean_max"),
            )
            for metric_group, metric_name, gate_name in gate_checks:
                try:
                    gate_value = float(gates[gate_name])
                except (KeyError, TypeError, ValueError) as exc:
                    raise ValueError(
                        f"validated calibration profile requires numeric gate {gate_name}"
                    ) from exc
                if not np.isfinite(gate_value) or gate_value < 0:
                    raise ValueError(f"quality gate {gate_name} must be non-negative")
                if float(quality[metric_group][metric_name]) > gate_value:
                    raise ValueError(
                        f"quality.passed conflicts with {metric_group}.{metric_name}"
                    )
            if quality.get("failures") not in (None, []):
                raise ValueError("validated calibration profile cannot list quality failures")
            rank = quality.get("rank")
            condition = quality.get("condition_number")
            if not isinstance(rank, int) or rank != expected_shape[0]:
                raise ValueError("validated calibration profile quality.rank is inconsistent")
            try:
                condition_value = float(condition)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "validated calibration profile condition_number must be numeric"
                ) from exc
            if not np.isfinite(condition_value) or condition_value <= 0:
                raise ValueError(
                    "validated calibration profile condition_number must be finite and positive"
                )
            datasets = _require_mapping(data, "datasets")
            training = datasets.get("training")
            validation = datasets.get("validation")
            if not isinstance(training, dict) or not isinstance(validation, dict):
                raise ValueError(
                    "validated calibration profiles require training and validation datasets"
                )
            if validation.get("independent") is not True:
                raise ValueError("validation dataset must be declared independent")
            if validation.get("content_distinct_from_training") is not True:
                raise ValueError(
                    "validation dataset content must differ from training data"
                )
            for label, dataset in (("training", training), ("validation", validation)):
                if not _is_sha256(dataset.get("sha256")):
                    raise ValueError(f"{label} dataset requires a SHA-256 content hash")
                if not isinstance(dataset.get("patch_count"), int) or dataset["patch_count"] <= 0:
                    raise ValueError(f"{label} dataset patch_count must be positive")
            if training["sha256"].lower() == validation["sha256"].lower():
                raise ValueError("training and validation dataset hashes must differ")

        integrity = _require_mapping(data, "integrity")
        stored_hash = integrity.get("sha256")
        if not _is_sha256(stored_hash):
            raise ValueError("calibration profile integrity.sha256 must be a SHA-256 hex string")
        expected_hash = calibration_profile_sha256(data)
        if verify_integrity and stored_hash.lower() != expected_hash:
            raise ValueError(
                "calibration profile SHA-256 mismatch: file content has changed"
            )
        integrity["sha256"] = expected_hash
        model["matrix"] = matrix.tolist()
        return cls(data)

    @property
    def profile_id(self) -> str:
        return self.data["profile_id"]

    @property
    def status(self) -> str:
        return self.data["status"]

    @property
    def sha256(self) -> str:
        return self.data["integrity"]["sha256"]

    @property
    def input_domain(self) -> str:
        return self.data["input"]["domain"]

    @property
    def model_type(self) -> str:
        return self.data["model"]["type"]

    @property
    def degree(self) -> int:
        if self.model_type == "linear_3x3":
            return 1
        if self.model_type == "root_polynomial_2":
            return 2
        return int(self.data["model"]["degree"])

    @property
    def matrix(self) -> np.ndarray:
        return np.asarray(self.data["model"]["matrix"], dtype=np.float64).copy()

    def to_dict(self) -> Dict[str, Any]:
        return deepcopy(self.data)


def write_calibration_profile(profile: Mapping[str, Any], path: str | Path) -> Path:
    """Canonicalize, hash, validate, and write a CCM profile as UTF-8 JSON."""
    data = deepcopy(dict(profile))
    data.setdefault("integrity", {})["sha256"] = calibration_profile_sha256(data)
    validated = CalibrationProfile.from_mapping(data)
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(validated.to_dict(), indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    return output_path


def load_calibration_profile(path: str | Path) -> CalibrationProfile:
    """Load and verify a versioned CCM profile."""
    profile_path = Path(path)
    if not profile_path.is_file():
        raise FileNotFoundError(
            f"Color calibration profile does not exist: {profile_path}"
        )
    with profile_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError("calibration profile root must be a JSON object")
    return CalibrationProfile.from_mapping(data)
