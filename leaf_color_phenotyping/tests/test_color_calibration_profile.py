import json

import numpy as np
import pytest

from src.color_calibration import (
    CalibrationModelFit,
    apply_calibration_profile,
    calibration_profile_sha256,
    fit_rgb_to_xyz_model,
    lab_d50_to_xyz_d65,
    load_calibration_profile,
    root_polynomial_expansion_2,
    validate_rgb_to_xyz_model,
    write_calibration_profile,
)
from src.pipeline import LeafColorPipeline
from src.utils import (
    COLORCHECKER_24_PATCH_IDS,
    get_colorchecker_reference_lab,
    write_image_rgb,
    xyz_to_lab,
)
from scripts.calibrate_color import choose_model, main as calibration_main, read_patch_csv


def make_profile(
    *,
    status="validated",
    matrix=None,
    input_domain="encoded_srgb",
    target_space="sRGB",
    model_type="linear_3x3",
    input_kind="jpeg",
    white_balance="none",
    raw_use_camera_wb=False,
):
    if matrix is None:
        matrix = np.eye(3).tolist() if model_type == "linear_3x3" else np.zeros((6, 3)).tolist()
    quality = {
        "passed": True,
        "rank": 6 if model_type == "root_polynomial_2" else 3,
        "condition_number": 1.0,
        "training_delta_e00": {
            "mean": 0.4,
            "median": 0.3,
            "p95": 0.9,
            "max": 1.0,
        },
        "validation_delta_e00": {
            "mean": 0.5,
            "median": 0.4,
            "p95": 1.0,
            "max": 1.2,
        },
        "vegetation_validation_delta_e00": {
            "mean": 0.5,
            "median": 0.4,
            "p95": 1.0,
            "max": 1.2,
        },
        "neutral_validation_delta_e00": {
            "mean": 0.5,
            "median": 0.4,
            "p95": 1.0,
            "max": 1.2,
        },
        "gates": {
            "median_max": 2.5,
            "p95_max": 6.0,
            "vegetation_mean_max": 3.0,
            "neutral_mean_max": 2.0,
        },
        "failures": [],
    }
    return {
        "schema_version": 2,
        "profile_id": "test-camera-d65",
        "status": status,
        "created_at": "2026-01-01T00:00:00+00:00",
        "input": {
            "kind": input_kind,
            "domain": input_domain,
            "range": [0.0, 1.0],
            "source_scale": "1",
            "channel_order": "RGB",
            "camera_id": "test-camera",
        },
        "preprocessing": {
            "white_balance": white_balance,
            "exposure_normalization": "fixed_capture",
            "raw_use_camera_wb": raw_use_camera_wb,
        },
        "model": {
            "type": model_type,
            "degree": 2 if model_type == "root_polynomial_2" else 1,
            "matrix_layout": "features_by_output",
            "matrix": matrix,
        },
        "target": {
            "space": target_space,
            "illuminant": "D65",
            "observer": "2_degree",
        },
        "reference": {
            "chart": "ColorChecker Classic 24",
            "id": "after_nov_2014",
            "source": "X-Rite 2016",
        },
        "datasets": {
            "training": {
                "path": "training.csv",
                "sha256": "0" * 64,
                "patch_count": 24,
            },
            "validation": {
                "path": "validation.csv",
                "sha256": "1" * 64,
                "patch_count": 24,
                "independent": True,
                "content_distinct_from_training": True,
            },
        },
        "quality": quality,
        "integrity": {},
    }


def test_profile_round_trip_has_stable_content_hash(tmp_path):
    path = write_calibration_profile(make_profile(), tmp_path / "camera.ccm.json")

    loaded = load_calibration_profile(path)

    assert loaded.profile_id == "test-camera-d65"
    assert loaded.status == "validated"
    assert loaded.input_domain == "encoded_srgb"
    assert np.array_equal(loaded.matrix, np.eye(3))
    assert loaded.sha256 == calibration_profile_sha256(loaded.to_dict())


def test_profile_hash_detects_matrix_replacement(tmp_path):
    path = write_calibration_profile(make_profile(), tmp_path / "camera.ccm.json")
    data = json.loads(path.read_text(encoding="utf-8"))
    data["model"]["matrix"][0][0] = 0.5
    path.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        load_calibration_profile(path)


def test_validated_profile_cannot_claim_failed_quality_metrics(tmp_path):
    profile = make_profile()
    profile["quality"]["validation_delta_e00"]["median"] = 3.0

    with pytest.raises(ValueError, match="quality.passed conflicts"):
        write_calibration_profile(profile, tmp_path / "invalid.ccm.json")


def test_required_mode_loads_only_validated_profile(tmp_path):
    write_calibration_profile(make_profile(), tmp_path / "camera.ccm.json")
    pipeline = LeafColorPipeline({
        "_config_dir": str(tmp_path),
        "imaging": {"camera_id": "test-camera"},
        "color_calibration": {
            "mode": "required",
            "profile_file": "camera.ccm.json",
        },
    })

    manifest = pipeline._color_calibration_manifest()

    assert pipeline.preprocessor.has_color_correction_matrix
    assert pipeline.color_calibration_status == "applied_validated_profile"
    assert manifest["profile_id"] == "test-camera-d65"
    assert manifest["profile_sha256"]
    assert manifest["matrix"] == np.eye(3).tolist()
    assert manifest["preprocessing"]["white_balance"] == "none"
    assert manifest["datasets"]["validation"]["independent"] is True


def test_required_mode_rejects_draft_profile(tmp_path):
    write_calibration_profile(
        make_profile(status="draft"), tmp_path / "draft.ccm.json"
    )

    with pytest.raises(ValueError, match="validated profile"):
        LeafColorPipeline({
            "_config_dir": str(tmp_path),
            "color_calibration": {
                "mode": "required",
                "profile_file": "draft.ccm.json",
            },
        })


def test_draft_profile_cannot_be_applied_directly(tmp_path):
    path = write_calibration_profile(
        make_profile(status="draft"), tmp_path / "draft.ccm.json"
    )

    with pytest.raises(ValueError, match="Only validated"):
        apply_calibration_profile(
            load_calibration_profile(path),
            np.full((1, 1, 3), 0.5, dtype=np.float32),
        )


def test_required_mode_rejects_profile_preprocessing_mismatch(tmp_path):
    write_calibration_profile(make_profile(), tmp_path / "camera.ccm.json")

    with pytest.raises(ValueError, match="preprocessing does not match"):
        LeafColorPipeline({
            "_config_dir": str(tmp_path),
            "imaging": {
                "camera_id": "test-camera",
                "white_balance": "gray_world",
                "exposure_normalization": "fixed_capture",
            },
            "color_calibration": {
                "mode": "required",
                "profile_file": "camera.ccm.json",
            },
        })


def test_required_mode_rejects_profile_camera_mismatch(tmp_path):
    write_calibration_profile(make_profile(), tmp_path / "camera.ccm.json")

    with pytest.raises(ValueError, match="camera_id"):
        LeafColorPipeline({
            "_config_dir": str(tmp_path),
            "imaging": {"camera_id": "another-camera"},
            "color_calibration": {
                "mode": "required",
                "profile_file": "camera.ccm.json",
            },
        })


def test_gray_card_profile_requires_exact_rgb_metadata(tmp_path):
    profile = make_profile(white_balance="gray_card")

    with pytest.raises(ValueError, match="gray_card_rgb"):
        write_calibration_profile(profile, tmp_path / "gray-card.ccm.json")


def test_required_mode_rejects_legacy_bare_matrix():
    with pytest.raises(ValueError, match="legacy bare CCM"):
        LeafColorPipeline({
            "color_calibration": {
                "mode": "required",
                "allow_legacy_matrix": True,
                "method": "linear",
                "matrix": np.eye(3).tolist(),
            }
        })


def test_colorchecker_references_are_explicit_and_distinct():
    before = get_colorchecker_reference_lab("before_nov_2014")
    after = get_colorchecker_reference_lab("after_nov_2014")

    assert before.shape == after.shape == (24, 3)
    assert not np.array_equal(before, after)
    before[0, 0] = -1
    assert get_colorchecker_reference_lab("before_nov_2014")[0, 0] >= 0


def test_root_polynomial_basis_scales_linearly_with_exposure():
    rgb = np.array([[0.2, 0.4, 0.6], [0.7, 0.3, 0.1]])

    assert np.allclose(
        root_polynomial_expansion_2(rgb * 0.5),
        root_polynomial_expansion_2(rgb) * 0.5,
    )


def test_linear_rgb_to_xyz_fit_recovers_known_matrix():
    rng = np.random.default_rng(11)
    measured = rng.uniform(0.05, 0.9, size=(24, 3))
    rgb_to_xyz_matrix = np.array([
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ]).T
    target = measured @ rgb_to_xyz_matrix

    fit = fit_rgb_to_xyz_model(measured, target, model_type="linear_3x3")
    metrics = validate_rgb_to_xyz_model(measured, target, fit)

    assert np.allclose(fit.matrix, rgb_to_xyz_matrix)
    assert metrics["max"] < 1e-10


def test_reference_lab_is_converted_to_unclipped_xyz():
    reference = get_colorchecker_reference_lab("before_nov_2014")

    xyz = lab_d50_to_xyz_d65(reference)

    assert xyz.shape == (24, 3)
    assert np.isfinite(xyz).all()
    assert xyz[17, 0] > 0


def test_xyz_profile_application_reports_display_gamut_clipping(tmp_path):
    exaggerated = np.array([
        [2.0, 0.0, 0.0],
        [0.0, 2.0, 0.0],
        [0.0, 0.0, 2.0],
    ])
    path = write_calibration_profile(
        make_profile(
            matrix=exaggerated.tolist(),
            input_domain="linear_srgb",
            target_space="XYZ",
        ),
        tmp_path / "linear.ccm.json",
    )
    profile = load_calibration_profile(path)

    result = apply_calibration_profile(
        profile, np.full((2, 2, 3), 0.8, dtype=np.float32)
    )

    assert result.xyz_d65.shape == (2, 2, 3)
    assert result.qc["QC_CCM_clipped_fraction"] == 1.0
    assert np.all((result.srgb >= 0) & (result.srgb <= 1))


def test_profile_clipping_qc_is_computed_only_inside_mask(tmp_path):
    path = write_calibration_profile(
        make_profile(
            matrix=(np.eye(3) * 2).tolist(),
            input_domain="linear_srgb",
            target_space="XYZ",
        ),
        tmp_path / "masked.ccm.json",
    )
    image = np.full((2, 2, 3), 0.8, dtype=np.float32)
    image[0, 0] = 0
    mask = np.zeros((2, 2), dtype=np.uint8)
    mask[0, 0] = 255

    result = apply_calibration_profile(
        load_calibration_profile(path), image, mask=mask
    )

    assert result.qc["QC_CCM_clipped_fraction"] == 0.0


def test_xyz_to_lab_keeps_negative_out_of_gamut_values_finite():
    lab = xyz_to_lab(np.array([[-0.02, 0.1, -0.01]], dtype=np.float64))

    assert np.isfinite(lab).all()


def test_linear_xyz_profile_preserves_segmentation_and_marks_qc(tmp_path):
    image = np.full((80, 80, 3), 255, dtype=np.uint8)
    image[20:60, 25:55] = [30, 160, 30]
    image_path = tmp_path / "leaf.png"
    write_image_rgb(image_path, image)
    rgb_to_xyz_matrix = np.array([
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ]).T
    write_calibration_profile(
        make_profile(
            matrix=rgb_to_xyz_matrix.tolist(),
            input_domain="linear_srgb",
            target_space="XYZ",
            white_balance="gray_world",
        ),
        tmp_path / "linear.ccm.json",
    )
    common = {
        "imaging": {
            "camera_id": "test-camera",
            "white_balance": "gray_world",
            "exposure_normalization": "fixed_capture",
        },
        "segmentation": {
            "method": "exg",
            "min_leaf_area_ratio": 0.01,
            "exclude_white_tissue": False,
        },
        "features": {
            "vegetation_indices": {"enabled": False},
            "texture": {"enabled": False},
            "shape": {"enabled": False},
        },
    }
    baseline_pipeline = LeafColorPipeline(common)
    calibrated_pipeline = LeafColorPipeline({
        **common,
        "_config_dir": str(tmp_path),
        "color_calibration": {
            "mode": "required",
            "profile_file": "linear.ccm.json",
        },
    })

    baseline = baseline_pipeline.process_single(
        str(image_path), return_visualization=True, verbose=False
    )
    calibrated = calibrated_pipeline.process_single(
        str(image_path), return_visualization=True, verbose=False
    )

    assert np.array_equal(baseline["mask"], calibrated["mask"])
    assert calibrated["features"]["QC_CCM_applied"] == 1.0
    assert calibrated_pipeline.color_calibration_applied


def test_linear_xyz_profile_preserves_empty_mask_qc_row(tmp_path):
    image_path = tmp_path / "blank.png"
    write_image_rgb(image_path, np.zeros((32, 32, 3), dtype=np.float32))
    write_calibration_profile(
        make_profile(
            matrix=np.eye(3).tolist(),
            input_domain="linear_srgb",
            target_space="XYZ",
        ),
        tmp_path / "linear.ccm.json",
    )
    pipeline = LeafColorPipeline({
        "_config_dir": str(tmp_path),
        "imaging": {"camera_id": "test-camera"},
        "color_calibration": {
            "mode": "required",
            "profile_file": "linear.ccm.json",
        },
        "segmentation": {
            "method": "exg",
            "exclude_white_tissue": False,
        },
        "features": {
            "vegetation_indices": {"enabled": False},
            "texture": {"enabled": False},
            "shape": {"enabled": False},
        },
    })
    pipeline.segmenter.segment = lambda image: np.zeros(
        image.shape[:2], dtype=np.uint8
    )

    result = pipeline.process_single(str(image_path), verbose=False)

    assert result["features"]["QC_mask_is_empty"] == 1.0
    assert result["features"]["QC_CCM_applied"] == 1.0
    assert np.isnan(result["features"]["QC_CCM_clipped_fraction"])


def _write_patch_csv(path, rgb, *, scale=65535, decimals=8):
    lines = ["patch_id,R,G,B"]
    for patch_id, values in zip(COLORCHECKER_24_PATCH_IDS, rgb * scale):
        rendered = ",".join(f"{value:.{decimals}f}" for value in values)
        lines.append(f"{patch_id},{rendered}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _dummy_fit(model_type):
    terms = 3 if model_type == "linear_3x3" else 6
    return CalibrationModelFit(
        model_type=model_type,
        matrix=np.zeros((terms, 3)),
        rank=terms,
        condition_number=1.0,
        residuals=np.empty(0),
        training_delta_e00={"mean": 0.0, "median": 0.0, "p95": 0.0, "max": 0.0},
    )


def test_patch_csv_requires_explicit_scale_and_canonical_ids(tmp_path):
    rgb = np.linspace(0.05, 0.95, 72).reshape(24, 3)
    path = tmp_path / "patches.csv"
    _write_patch_csv(path, rgb, scale=255)

    patch_ids, loaded = read_patch_csv(path, input_scale="255")

    assert patch_ids == COLORCHECKER_24_PATCH_IDS
    assert np.allclose(loaded, rgb, atol=1e-8)
    with pytest.raises(ValueError, match=r"outside \[0, 1\]"):
        read_patch_csv(path, input_scale="1")


def test_auto_model_selection_requires_material_non_regressive_gain():
    fits = {
        "linear_3x3": _dummy_fit("linear_3x3"),
        "root_polynomial_2": _dummy_fit("root_polynomial_2"),
    }

    assert choose_model(fits, {
        "linear_3x3": {"median": 2.0, "p95": 4.0},
        "root_polynomial_2": {"median": 1.7, "p95": 4.0},
    }) == "root_polynomial_2"
    assert choose_model(fits, {
        "linear_3x3": {"median": 2.0, "p95": 4.0},
        "root_polynomial_2": {"median": 1.7, "p95": 4.1},
    }) == "linear_3x3"
    assert choose_model(fits, {
        "linear_3x3": {"median": 2.0, "p95": 4.0},
        "root_polynomial_2": {"median": 1.85, "p95": 3.0},
    }) == "linear_3x3"


def test_calibration_cli_fit_creates_validated_xyz_profile(tmp_path):
    target_xyz = lab_d50_to_xyz_d65(
        get_colorchecker_reference_lab("after_nov_2014")
    )
    measured = target_xyz / 1.2
    validation_measured = measured.copy()
    validation_measured[:, 0] *= 1.00002
    training_csv = tmp_path / "training.csv"
    validation_csv = tmp_path / "validation.csv"
    _write_patch_csv(training_csv, measured, decimals=8)
    _write_patch_csv(validation_csv, validation_measured, decimals=10)
    output = tmp_path / "synthetic.ccm.json"

    return_code = calibration_main([
        "fit",
        "--training-csv", str(training_csv),
        "--validation-csv", str(validation_csv),
        "--output", str(output),
        "--model", "linear_3x3",
        "--input-domain", "linear_srgb",
        "--profile-id", "synthetic-camera-d65",
        "--camera-id", "synthetic-camera",
        "--input-kind", "rendered_rgb",
        "--input-scale", "65535",
        "--white-balance", "none",
        "--exposure-normalization", "fixed-capture",
        "--no-raw-use-camera-wb",
        "--reference-id", "after_nov_2014",
    ])

    profile = load_calibration_profile(output)
    assert return_code == 0
    assert profile.status == "validated"
    assert profile.data["target"]["space"] == "XYZ"
    assert np.allclose(profile.matrix, np.eye(3) * 1.2, atol=1e-6)
    assert profile.data["quality"]["validation_delta_e00"]["max"] < 0.01
    assert calibration_main([
        "validate",
        "--profile", str(output),
        "--validation-csv", str(validation_csv),
        "--input-scale", "65535",
    ]) == 0


def test_legacy_migration_is_always_draft(tmp_path):
    matrix_path = tmp_path / "legacy.npy"
    np.save(matrix_path, np.eye(3))
    output = tmp_path / "legacy.ccm.json"

    return_code = calibration_main([
        "migrate-legacy",
        "--matrix", str(matrix_path),
        "--output", str(output),
        "--model", "linear_3x3",
        "--profile-id", "legacy-camera",
        "--camera-id", "camera-1",
        "--input-kind", "rendered_rgb",
        "--input-scale", "1",
        "--white-balance", "none",
        "--exposure-normalization", "unknown",
        "--no-raw-use-camera-wb",
        "--reference-id", "before_nov_2014",
    ])

    profile = load_calibration_profile(output)
    assert return_code == 0
    assert profile.status == "draft"
    assert profile.data["migration"]["source_sha256"]
