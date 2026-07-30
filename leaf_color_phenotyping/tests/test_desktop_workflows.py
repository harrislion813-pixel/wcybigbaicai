import json
from pathlib import Path

import numpy as np
import pytest

from application.analysis_service import AnalysisService
from application.calibration_service import CalibrationService
from application.models import AnalysisRequest, CalibrationImageRequest
from application.profile_registry import ProfileRegistry
from src.calibration_workflow import (
    CalibrationProfileRequest, fit_calibration_profile,
)
from src.color_calibration import (
    lab_d50_to_xyz_d65, load_calibration_profile, srgb_to_linear,
    xyz_d65_to_srgb,
)
from src.colorchecker_detection import extract_colorchecker_patches, full_image_corners
from src.pipeline import LeafColorPipeline
from src.utils import get_colorchecker_reference_lab, write_image_rgb


def _synthetic_chart() -> tuple[np.ndarray, np.ndarray]:
    height, width = 600, 900
    image = np.zeros((height, width, 3), dtype=np.float32)
    values = []
    for row in range(3):
        for column in range(6):
            value = np.array([
                0.12 + column * 0.09,
                0.18 + row * 0.20,
                0.78 - column * 0.07,
            ], dtype=np.float32)
            values.append(value)
            image[row * 150:(row + 1) * 150, column * 150:(column + 1) * 150] = value
    for column, level in enumerate(np.linspace(0.92, 0.08, 6)):
        value = np.full(3, level, dtype=np.float32)
        values.append(value)
        image[450:600, column * 150:(column + 1) * 150] = value
    return image, np.asarray(values)


def test_colorchecker_full_image_sampling_preserves_patch_order():
    chart, expected = _synthetic_chart()
    result = extract_colorchecker_patches(
        chart,
        display_rgb=chart,
        corners=full_image_corners(chart),
    )

    assert result.rgb.shape == (24, 3)
    assert np.allclose(result.rgb, expected, atol=1e-3)
    assert result.preview_rgb.shape == (600, 900, 3)


def test_colorchecker_auto_detects_a_framed_reference_chart():
    target_xyz = lab_d50_to_xyz_d65(
        get_colorchecker_reference_lab("after_nov_2014")
    )
    reference_srgb, _ = xyz_d65_to_srgb(target_xyz)
    reference_srgb = np.clip(reference_srgb, 0, 1).astype(np.float32)
    canvas = np.full((720, 1080, 3), 0.65, dtype=np.float32)
    chart = np.full((480, 720, 3), 0.025, dtype=np.float32)
    for index, color in enumerate(reference_srgb):
        row, column = divmod(index, 6)
        y1, y2 = row * 120 + 20, (row + 1) * 120 - 20
        x1, x2 = column * 120 + 20, (column + 1) * 120 - 20
        chart[y1:y2, x1:x2] = color
    canvas[120:600, 180:900] = chart

    result = extract_colorchecker_patches(
        srgb_to_linear(canvas).astype(np.float32), display_rgb=canvas
    )

    assert result.rgb.shape == (24, 3)
    assert np.mean(np.abs(result.rgb - srgb_to_linear(reference_srgb))) < 0.08


def test_image_workflow_can_create_a_validated_profile(tmp_path):
    training_source = tmp_path / "training.png"
    validation_source = tmp_path / "validation.png"
    training_source.write_bytes(b"independent-training-capture")
    validation_source.write_bytes(b"independent-validation-capture")
    target_xyz = lab_d50_to_xyz_d65(
        get_colorchecker_reference_lab("after_nov_2014")
    )
    training = target_xyz.copy()
    perturbation = np.linspace(-2e-4, 2e-4, target_xyz.size).reshape(target_xyz.shape)
    validation = np.clip(target_xyz + perturbation, 0, 1)

    result = fit_calibration_profile(CalibrationProfileRequest(
        training_rgb=training,
        validation_rgb=validation,
        training_source=training_source,
        validation_source=validation_source,
        output_path=tmp_path / "profile.ccm.json",
        profile_id="test-camera-v1",
        camera_id="test-camera",
        input_kind="rendered_rgb",
        reference_id="after_nov_2014",
    ))

    assert result.status == "validated"
    assert load_calibration_profile(result.profile_path).status == "validated"
    summaries = ProfileRegistry([tmp_path]).scan()
    assert len(summaries) == 1
    assert summaries[0].selectable


def test_image_workflow_rejects_duplicated_validation_patches(tmp_path):
    training_source = tmp_path / "training.png"
    validation_source = tmp_path / "validation.png"
    training_source.write_bytes(b"training-container")
    validation_source.write_bytes(b"different-container")
    patches = lab_d50_to_xyz_d65(
        get_colorchecker_reference_lab("after_nov_2014")
    )

    with pytest.raises(ValueError, match="patch measurements.*independent capture"):
        fit_calibration_profile(CalibrationProfileRequest(
            training_rgb=patches,
            validation_rgb=patches.copy(),
            training_source=training_source,
            validation_source=validation_source,
            output_path=tmp_path / "profile.ccm.json",
            profile_id="duplicate-validation",
            camera_id="test-camera",
            input_kind="rendered_rgb",
            reference_id="after_nov_2014",
        ))


def test_analysis_service_hides_ccm_for_relative_runs(tmp_path):
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    output_dir = tmp_path / "output"
    project_dir = Path(__file__).resolve().parent.parent
    service = AnalysisService(project_dir)

    config = service.build_config(AnalysisRequest(
        input_dir=image_dir,
        output_dir=output_dir,
        calibration_mode="relative",
    ))

    assert config["color_calibration"]["mode"] == "off"
    assert config["color_calibration"]["profile_file"] == ""
    assert config["segmentation"]["method"] == "auto"


def test_analysis_service_rejects_output_inside_input_tree(tmp_path):
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    service = AnalysisService(Path(__file__).resolve().parent.parent)

    with pytest.raises(ValueError, match="结果文件夹不能位于图片文件夹内部"):
        service.build_config(AnalysisRequest(
            input_dir=image_dir,
            output_dir=image_dir / "results",
        ))


def test_analysis_service_runs_without_cli_arguments(tmp_path):
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    image = np.full((80, 80, 3), 0.1, dtype=np.float32)
    image[20:60, 20:60] = [0.1, 0.8, 0.1]
    write_image_rgb(image_dir / "leaf.png", image)
    output_dir = tmp_path / "output"
    service = AnalysisService(Path(__file__).resolve().parent.parent)
    events = []

    result = service.run(AnalysisRequest(
        input_dir=image_dir,
        output_dir=output_dir,
        group_by_sample=False,
        save_visualizations=True,
        segmentation_method="exg",
    ), progress_callback=events.append)

    assert result.table_path.is_file()
    assert result.manifest_path is not None and result.manifest_path.is_file()
    assert result.visualization_dir is not None and result.visualization_dir.is_dir()
    assert any(event.status == "success" for event in events)


def test_calibration_service_builds_profile_directly_from_two_images(tmp_path):
    target_xyz = lab_d50_to_xyz_d65(
        get_colorchecker_reference_lab("after_nov_2014")
    )
    reference_srgb, _ = xyz_d65_to_srgb(target_xyz)
    reference_srgb = np.clip(reference_srgb, 0, 1).astype(np.float32)
    chart = np.zeros((600, 900, 3), dtype=np.float32)
    for index, color in enumerate(reference_srgb):
        row, column = divmod(index, 6)
        chart[row * 150:(row + 1) * 150, column * 150:(column + 1) * 150] = color
    training_path = tmp_path / "training.png"
    validation_path = tmp_path / "validation.png"
    write_image_rgb(training_path, chart)
    validation_chart = np.clip(chart * 0.995 + 0.0005, 0, 1)
    write_image_rgb(validation_path, validation_chart)
    corners = full_image_corners(chart)

    result = CalibrationService().create_profile(CalibrationImageRequest(
        training_image=training_path,
        validation_image=validation_path,
        output_path=tmp_path / "image-profile.ccm.json",
        profile_id="image-profile-v1",
        camera_id="camera-a",
        training_corners=corners,
        validation_corners=corners,
    ))

    assert result.profile_path.is_file()
    assert result.training_patch_csv.is_file()
    assert result.validation_patch_csv.is_file()
    assert result.training_preview.shape == (600, 900, 3)


def test_batch_cancel_saves_partial_result_and_manifest(tmp_path):
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    for index in range(2):
        image = np.full((80, 80, 3), 0.1, dtype=np.float32)
        image[20:60, 20:60] = [0.1, 0.8, 0.1]
        write_image_rgb(image_dir / f"leaf_{index}.png", image)
    output = tmp_path / "result.csv"
    pipeline = LeafColorPipeline({
        "segmentation": {
            "method": "exg", "morph_kernel_size": 1,
            "min_leaf_area_ratio": 0.001,
        },
        "features": {
            "texture": {"enabled": False},
            "shape": {"enabled": False},
            "vegetation_indices": {"enabled": False},
        },
    })
    cancel = {"requested": False}

    def progress(event):
        if event["status"] == "success":
            cancel["requested"] = True

    result = pipeline.process_batch(
        str(image_dir), output_csv=str(output), group_by_sample=False,
        verbose=False, progress_callback=progress,
        cancel_check=lambda: cancel["requested"],
    )

    assert len(result) == 1
    assert pipeline.last_batch_cancelled
    manifest = json.loads(
        (tmp_path / "result_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["status"] == "cancelled"
    assert manifest["cancelled"] is True
