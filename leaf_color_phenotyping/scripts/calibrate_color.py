#!/usr/bin/env python3
"""Create, validate, inspect, and migrate color-calibration profiles.

Measured patch CSV files must contain exactly these columns:
    patch_id,R,G,B

RGB scale is never inferred. Pass --input-scale explicitly so a 0-255 table
cannot silently produce a matrix that blackens normalized images.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Dict, Iterable, Mapping

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.color_calibration import (  # noqa: E402
    CalibrationModelFit,
    fit_rgb_to_xyz_model,
    lab_d50_to_xyz_d65,
    load_calibration_profile,
    rgb_to_xyz_delta_e00,
    validate_rgb_to_xyz_model,
    write_calibration_profile,
    xyz_d65_to_srgb,
)
from src.calibration_workflow import (  # noqa: E402
    CalibrationGates,
    CalibrationProfileRequest,
    fit_calibration_profile,
)
from src.utils import (  # noqa: E402
    COLORCHECKER_24_PATCH_IDS,
    get_colorchecker_reference_lab,
)


INPUT_SCALES = {"1": 1.0, "255": 255.0, "65535": 65535.0}
VEGETATION_PATCH_IDS = {
    "foliage",
    "bluish_green",
    "yellow_green",
    "green",
    "cyan",
}
NEUTRAL_PATCH_IDS = {
    "white_95",
    "neutral_8",
    "neutral_65",
    "neutral_5",
    "neutral_35",
    "black_2",
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _metrics(values: np.ndarray) -> Dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or len(array) == 0 or not np.isfinite(array).all():
        raise ValueError("Delta E values must be a non-empty finite vector")
    return {
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p95": float(np.percentile(array, 95)),
        "max": float(np.max(array)),
    }


def read_patch_csv(
    path: str | Path,
    *,
    input_scale: str,
    required_patch_ids: Iterable[str] = COLORCHECKER_24_PATCH_IDS,
) -> tuple[tuple[str, ...], np.ndarray]:
    """Read, normalize, and deterministically order a measured patch table."""
    csv_path = Path(path)
    if not csv_path.is_file():
        raise FileNotFoundError(f"Patch CSV does not exist: {csv_path}")
    if input_scale not in INPUT_SCALES:
        raise ValueError(f"input_scale must be one of {sorted(INPUT_SCALES)}")

    required = tuple(required_patch_ids)
    expected = {"patch_id", "R", "G", "B"}
    rows: Dict[str, list[float]] = {}
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or set(reader.fieldnames) != expected:
            raise ValueError(
                "Patch CSV columns must be exactly: patch_id,R,G,B"
            )
        for line_number, row in enumerate(reader, start=2):
            patch_id = (row.get("patch_id") or "").strip()
            if not patch_id:
                raise ValueError(f"Empty patch_id at CSV line {line_number}")
            if patch_id in rows:
                raise ValueError(f"Duplicate patch_id '{patch_id}'")
            try:
                values = [float(row[channel]) for channel in ("R", "G", "B")]
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Non-numeric RGB value for patch '{patch_id}'"
                ) from exc
            rows[patch_id] = values

    missing = sorted(set(required) - set(rows))
    unknown = sorted(set(rows) - set(required))
    if missing or unknown:
        details = []
        if missing:
            details.append(f"missing={missing}")
        if unknown:
            details.append(f"unknown={unknown}")
        raise ValueError("Patch IDs do not match the selected chart: " + "; ".join(details))

    rgb = np.asarray([rows[patch_id] for patch_id in required], dtype=np.float64)
    if not np.isfinite(rgb).all():
        raise ValueError("Patch CSV contains NaN or infinite RGB values")
    rgb /= INPUT_SCALES[input_scale]
    if np.any(rgb < 0) or np.any(rgb > 1):
        low = float(np.min(rgb))
        high = float(np.max(rgb))
        raise ValueError(
            f"RGB values are outside [0, 1] after --input-scale {input_scale}: "
            f"min={low:.6g}, max={high:.6g}"
        )
    return required, rgb


def choose_model(
    fits: Mapping[str, CalibrationModelFit],
    validation_metrics: Mapping[str, Mapping[str, float]],
    *,
    root_min_median_improvement: float = 0.10,
) -> str:
    """Prefer root-polynomial only for a material, non-regressive validation gain."""
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


def _fit_candidates(
    training_rgb: np.ndarray,
    validation_rgb: np.ndarray,
    target_xyz: np.ndarray,
    requested_model: str,
) -> tuple[CalibrationModelFit, Dict[str, Dict[str, float]], str]:
    model_types = (
        ("linear_3x3", "root_polynomial_2")
        if requested_model == "auto"
        else (requested_model,)
    )
    fits: Dict[str, CalibrationModelFit] = {}
    validation: Dict[str, Dict[str, float]] = {}
    for model_type in model_types:
        fit = fit_rgb_to_xyz_model(
            training_rgb,
            target_xyz,
            model_type=model_type,
        )
        fits[model_type] = fit
        validation[model_type] = validate_rgb_to_xyz_model(
            validation_rgb,
            target_xyz,
            fit,
        )
    selected = choose_model(fits, validation)
    return fits[selected], validation, selected


def _quality_report(
    fit: CalibrationModelFit,
    validation_rgb: np.ndarray,
    target_xyz: np.ndarray,
    patch_ids: tuple[str, ...],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    delta_e = rgb_to_xyz_delta_e00(validation_rgb, target_xyz, fit)
    id_to_index = {patch_id: index for index, patch_id in enumerate(patch_ids)}
    vegetation = delta_e[[id_to_index[patch_id] for patch_id in VEGETATION_PATCH_IDS]]
    neutral = delta_e[[id_to_index[patch_id] for patch_id in NEUTRAL_PATCH_IDS]]
    validation = _metrics(delta_e)
    vegetation_metrics = _metrics(vegetation)
    neutral_metrics = _metrics(neutral)
    gates = {
        "median_max": float(args.max_median_delta_e),
        "p95_max": float(args.max_p95_delta_e),
        "vegetation_mean_max": float(args.max_vegetation_mean_delta_e),
        "neutral_mean_max": float(args.max_neutral_mean_delta_e),
    }
    failures = []
    if validation["median"] > gates["median_max"]:
        failures.append("validation median Delta E 00")
    if validation["p95"] > gates["p95_max"]:
        failures.append("validation p95 Delta E 00")
    if vegetation_metrics["mean"] > gates["vegetation_mean_max"]:
        failures.append("vegetation-patch mean Delta E 00")
    if neutral_metrics["mean"] > gates["neutral_mean_max"]:
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
        "gates": gates,
        "validation_patch_delta_e00": {
            patch_id: float(delta_e[index])
            for index, patch_id in enumerate(patch_ids)
        },
        "outlier_patch_ids": [
            patch_id
            for index, patch_id in enumerate(patch_ids)
            if delta_e[index] > gates["p95_max"]
        ],
    }


def _reference_metadata(reference_id: str) -> Dict[str, str]:
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


def command_fit(args: argparse.Namespace) -> int:
    gate_values = (
        args.max_median_delta_e,
        args.max_p95_delta_e,
        args.max_vegetation_mean_delta_e,
        args.max_neutral_mean_delta_e,
    )
    if not np.isfinite(gate_values).all() or np.any(np.asarray(gate_values) < 0):
        raise ValueError("Delta E quality gates must be finite and non-negative")
    if args.white_balance == "gray_card" and args.gray_card_rgb is None:
        raise ValueError("--gray-card-rgb is required with --white-balance gray_card")
    if args.white_balance != "gray_card" and args.gray_card_rgb is not None:
        raise ValueError("--gray-card-rgb is only valid with --white-balance gray_card")
    if args.input_domain == "camera_linear_rgb" and args.input_kind != "raw":
        raise ValueError("camera_linear_rgb profiles require --input-kind raw")
    training_path = Path(args.training_csv).resolve()
    validation_path = Path(args.validation_csv).resolve()
    if training_path == validation_path:
        raise ValueError("Training and validation CSV files must be independent")
    training_sha256 = _sha256_file(training_path)
    validation_sha256 = _sha256_file(validation_path)
    if training_sha256 == validation_sha256:
        raise ValueError(
            "Training and validation CSV contents are identical; provide an "
            "independent capture"
        )

    patch_ids, training_rgb = read_patch_csv(
        training_path,
        input_scale=args.input_scale,
    )
    validation_ids, validation_rgb = read_patch_csv(
        validation_path,
        input_scale=args.input_scale,
    )
    if patch_ids != validation_ids:
        raise ValueError("Training and validation patch orders do not match")

    result = fit_calibration_profile(CalibrationProfileRequest(
        training_rgb=training_rgb,
        validation_rgb=validation_rgb,
        training_source=training_path,
        validation_source=validation_path,
        output_path=Path(args.output),
        profile_id=args.profile_id,
        camera_id=args.camera_id,
        input_kind=args.input_kind,
        reference_id=args.reference_id,
        input_domain=args.input_domain,
        white_balance=args.white_balance,
        exposure_normalization=args.exposure_normalization,
        raw_use_camera_wb=args.raw_use_camera_wb,
        gray_card_rgb=(
            tuple(float(value) for value in args.gray_card_rgb)
            if args.gray_card_rgb is not None else None
        ),
        model=args.model,
        source_scale=args.input_scale,
        gates=CalibrationGates(
            median_max=args.max_median_delta_e,
            p95_max=args.max_p95_delta_e,
            vegetation_mean_max=args.max_vegetation_mean_delta_e,
            neutral_mean_max=args.max_neutral_mean_delta_e,
        ),
    ))
    print(json.dumps({
        "profile": str(result.profile_path),
        "profile_id": args.profile_id,
        "status": result.status,
        "selected_model": result.selected_model,
        "validation_delta_e00": result.quality["validation_delta_e00"],
        "quality_failures": result.quality["failures"],
    }, indent=2, ensure_ascii=False))
    return 0 if result.status == "validated" else 2


def _profile_as_fit(profile: Any) -> CalibrationModelFit:
    if profile.data["target"]["space"] != "XYZ":
        raise ValueError("Validation requires a profile whose target space is XYZ")
    if profile.model_type not in {"linear_3x3", "root_polynomial_2"}:
        raise ValueError("Validation supports linear_3x3 and root_polynomial_2 profiles")
    quality = profile.data.get("quality", {})
    return CalibrationModelFit(
        model_type=profile.model_type,
        matrix=profile.matrix,
        rank=int(quality.get("rank", profile.matrix.shape[0])),
        condition_number=float(quality.get("condition_number", np.nan)),
        residuals=np.empty(0, dtype=np.float64),
        training_delta_e00=dict(quality.get("training_delta_e00", {})),
    )


def command_validate(args: argparse.Namespace) -> int:
    profile = load_calibration_profile(args.profile)
    reference_id = profile.data["reference"]["id"]
    if reference_id not in {"before_nov_2014", "after_nov_2014"}:
        raise ValueError("Custom references require a dedicated validation implementation")
    validation_path = Path(args.validation_csv).resolve()
    validation_sha256 = _sha256_file(validation_path)
    training = profile.data.get("datasets", {}).get("training", {})
    if validation_sha256 == training.get("sha256"):
        raise ValueError("Validation CSV is the profile training dataset")
    patch_ids, rgb = read_patch_csv(
        validation_path,
        input_scale=args.input_scale,
    )
    target_xyz = lab_d50_to_xyz_d65(get_colorchecker_reference_lab(reference_id))
    delta_e = rgb_to_xyz_delta_e00(rgb, target_xyz, _profile_as_fit(profile))
    id_to_index = {patch_id: index for index, patch_id in enumerate(patch_ids)}
    validation_metrics = _metrics(delta_e)
    vegetation_metrics = _metrics(
        delta_e[[id_to_index[item] for item in VEGETATION_PATCH_IDS]]
    )
    neutral_metrics = _metrics(
        delta_e[[id_to_index[item] for item in NEUTRAL_PATCH_IDS]]
    )
    gates = profile.data.get("quality", {}).get("gates")
    if not isinstance(gates, dict):
        raise ValueError("Profile does not define validation quality gates")
    failures = []
    if validation_metrics["median"] > float(gates["median_max"]):
        failures.append("validation median Delta E 00")
    if validation_metrics["p95"] > float(gates["p95_max"]):
        failures.append("validation p95 Delta E 00")
    if vegetation_metrics["mean"] > float(gates["vegetation_mean_max"]):
        failures.append("vegetation-patch mean Delta E 00")
    if neutral_metrics["mean"] > float(gates["neutral_mean_max"]):
        failures.append("neutral-patch mean Delta E 00")
    report = {
        "profile_id": profile.profile_id,
        "profile_status": profile.status,
        "validation_csv_sha256": validation_sha256,
        "passed": not failures,
        "failures": failures,
        "gates": gates,
        "validation_delta_e00": validation_metrics,
        "vegetation_validation_delta_e00": vegetation_metrics,
        "neutral_validation_delta_e00": neutral_metrics,
        "patch_delta_e00": {
            patch_id: float(delta_e[index])
            for index, patch_id in enumerate(patch_ids)
        },
    }
    rendered = json.dumps(report, indent=2, ensure_ascii=False)
    print(rendered)
    if args.output:
        Path(args.output).write_text(rendered + "\n", encoding="utf-8")
    return 0 if not failures else 2


def command_inspect(args: argparse.Namespace) -> int:
    profile = load_calibration_profile(args.profile)
    print(json.dumps({
        "profile_id": profile.profile_id,
        "status": profile.status,
        "sha256": profile.sha256,
        "input": profile.data["input"],
        "preprocessing": profile.data.get("preprocessing", {}),
        "model": {
            "type": profile.model_type,
            "matrix_shape": list(profile.matrix.shape),
        },
        "target": profile.data["target"],
        "reference": profile.data["reference"],
        "quality": profile.data["quality"],
    }, indent=2, ensure_ascii=False))
    return 0


def _read_legacy_matrix(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"Legacy matrix does not exist: {path}")
    suffix = path.suffix.lower()
    if suffix == ".npy":
        matrix = np.load(path, allow_pickle=False)
    elif suffix == ".json":
        matrix = np.asarray(json.loads(path.read_text(encoding="utf-8")), dtype=np.float64)
    elif suffix in {".csv", ".txt"}:
        matrix = np.loadtxt(path, delimiter="," if suffix == ".csv" else None)
    else:
        raise ValueError("Legacy matrix must be .npy, .json, .csv, or .txt")
    matrix = np.asarray(matrix, dtype=np.float64)
    if not np.isfinite(matrix).all():
        raise ValueError("Legacy matrix contains NaN or infinite values")
    return matrix


def command_migrate_legacy(args: argparse.Namespace) -> int:
    if args.white_balance == "gray_card" and args.gray_card_rgb is None:
        raise ValueError("--gray-card-rgb is required with --white-balance gray_card")
    if args.white_balance != "gray_card" and args.gray_card_rgb is not None:
        raise ValueError("--gray-card-rgb is only valid with --white-balance gray_card")
    source = Path(args.matrix).resolve()
    matrix = _read_legacy_matrix(source)
    if args.model == "linear_3x3":
        expected_shape = (3, 3)
        degree = 1
    else:
        terms = {1: 3, 2: 9, 3: 19}[args.degree]
        expected_shape = (terms, 3)
        degree = args.degree
    if matrix.shape != expected_shape:
        raise ValueError(
            f"Legacy matrix shape {matrix.shape} does not match {args.model}; "
            f"expected {expected_shape}"
        )

    profile = {
        "schema_version": 2,
        "profile_id": args.profile_id,
        "status": "draft",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "input": {
            "kind": args.input_kind,
            "domain": "encoded_srgb",
            "range": [0.0, 1.0],
            "source_scale": args.input_scale,
            "channel_order": "RGB",
            "camera_id": args.camera_id,
        },
        "preprocessing": {
            "white_balance": args.white_balance,
            "exposure_normalization": args.exposure_normalization,
            "raw_use_camera_wb": args.raw_use_camera_wb,
            "gray_card_rgb": args.gray_card_rgb,
        },
        "model": {
            "type": args.model,
            "degree": degree,
            "matrix_layout": "features_by_output",
            "matrix": matrix.tolist(),
        },
        "target": {
            "space": "sRGB",
            "illuminant": "D65",
            "observer": "2_degree",
        },
        "reference": _reference_metadata(args.reference_id),
        "migration": {
            "source_path": str(source),
            "source_sha256": _sha256_file(source),
            "warning": "Unvalidated legacy matrix; refit and independently validate before use",
        },
        "quality": {
            "passed": False,
            "failures": ["independent validation not available"],
        },
        "integrity": {},
    }
    output = write_calibration_profile(profile, args.output)
    print(json.dumps({
        "profile": str(output.resolve()),
        "status": "draft",
        "warning": "Draft profiles are rejected by required mode and are not applied by optional mode",
    }, indent=2, ensure_ascii=False))
    return 0


def _add_capture_metadata(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--profile-id", required=True)
    parser.add_argument("--camera-id", required=True)
    parser.add_argument(
        "--input-kind", choices=["raw", "rendered_rgb"], required=True
    )
    parser.add_argument("--input-scale", choices=sorted(INPUT_SCALES), required=True)
    parser.add_argument(
        "--white-balance",
        choices=["none", "gray_world", "perfect_reflector", "gray_card"],
        required=True,
    )
    parser.add_argument("--exposure-normalization", required=True)
    parser.add_argument(
        "--gray-card-rgb", nargs=3, type=float, metavar=("R", "G", "B")
    )
    raw_wb = parser.add_mutually_exclusive_group(required=True)
    raw_wb.add_argument(
        "--raw-use-camera-wb", dest="raw_use_camera_wb", action="store_true"
    )
    raw_wb.add_argument(
        "--no-raw-use-camera-wb", dest="raw_use_camera_wb", action="store_false"
    )
    parser.add_argument(
        "--reference-id",
        choices=["before_nov_2014", "after_nov_2014"],
        required=True,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build and audit version-2 color-calibration profiles",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    fit_parser = subparsers.add_parser("fit", help="fit and independently validate a profile")
    fit_parser.add_argument("--training-csv", required=True)
    fit_parser.add_argument("--validation-csv", required=True)
    fit_parser.add_argument("--output", required=True)
    fit_parser.add_argument(
        "--model",
        choices=["auto", "linear_3x3", "root_polynomial_2"],
        default="auto",
    )
    fit_parser.add_argument(
        "--input-domain",
        choices=["linear_srgb", "camera_linear_rgb"],
        required=True,
    )
    fit_parser.add_argument("--max-median-delta-e", type=float, default=2.5)
    fit_parser.add_argument("--max-p95-delta-e", type=float, default=6.0)
    fit_parser.add_argument("--max-vegetation-mean-delta-e", type=float, default=3.0)
    fit_parser.add_argument("--max-neutral-mean-delta-e", type=float, default=2.0)
    _add_capture_metadata(fit_parser)
    fit_parser.set_defaults(func=command_fit)

    validate_parser = subparsers.add_parser("validate", help="evaluate a profile on a patch CSV")
    validate_parser.add_argument("--profile", required=True)
    validate_parser.add_argument("--validation-csv", required=True)
    validate_parser.add_argument("--input-scale", choices=sorted(INPUT_SCALES), required=True)
    validate_parser.add_argument("--output")
    validate_parser.set_defaults(func=command_validate)

    inspect_parser = subparsers.add_parser("inspect", help="show profile provenance and quality")
    inspect_parser.add_argument("--profile", required=True)
    inspect_parser.set_defaults(func=command_inspect)

    migrate_parser = subparsers.add_parser(
        "migrate-legacy",
        help="wrap a bare legacy matrix as an explicitly unvalidated draft profile",
    )
    migrate_parser.add_argument("--matrix", required=True)
    migrate_parser.add_argument("--output", required=True)
    migrate_parser.add_argument(
        "--model",
        choices=["linear_3x3", "legacy_polynomial"],
        required=True,
    )
    migrate_parser.add_argument("--degree", type=int, choices=[1, 2, 3], default=1)
    _add_capture_metadata(migrate_parser)
    migrate_parser.set_defaults(func=command_migrate_legacy)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except (ValueError, FileNotFoundError, json.JSONDecodeError) as exc:
        parser.exit(1, f"error: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
