import json
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import src.segmentation as segmentation_module
from scripts import batch_extract
from src.pipeline import LeafColorPipeline
from src.preprocessing import ImagePreprocessor
from src.segmentation import BaseSegmenter, GrabCutSegmenter, UNetSegmenter
from src.utils import (
    COLORCHECKER_24_LAB_D50, find_images, parse_sample_id, read_image_rgb,
    split_pairs_by_sample, write_image_rgb,
)


def test_disabled_feature_groups_stay_disabled():
    pipeline = LeafColorPipeline({
        "features": {
            "vegetation_indices": {"enabled": False},
            "texture": {"enabled": False},
            "shape": {"enabled": False},
        }
    })

    assert pipeline.veg_index_extractor.indices == []
    assert pipeline.texture_extractor.properties == []
    assert pipeline.shape_extractor.features == []


def test_enabled_calibration_requires_a_matrix_source():
    with pytest.raises(ValueError, match="ccm_file"):
        LeafColorPipeline({"color_calibration": {"enabled": True}})


def test_inline_linear_ccm_is_validated_and_loaded():
    pipeline = LeafColorPipeline({
        "color_calibration": {
            "enabled": True,
            "method": "linear",
            "matrix": np.eye(3).tolist(),
        }
    })

    assert pipeline.preprocessor.has_color_correction_matrix


def test_relative_ccm_file_is_resolved_from_config_directory(tmp_path):
    matrix_path = tmp_path / "identity.npy"
    np.save(matrix_path, np.eye(3))

    pipeline = LeafColorPipeline({
        "_config_dir": str(tmp_path),
        "color_calibration": {
            "enabled": True,
            "method": "linear",
            "ccm_file": "identity.npy",
        },
    })

    assert pipeline.preprocessor.has_color_correction_matrix


def test_invalid_ccm_shape_is_rejected():
    preprocessor = ImagePreprocessor(calibration_method="linear")

    with pytest.raises(ValueError, match="CCM shape"):
        preprocessor.set_color_correction_matrix(np.eye(2))


def test_ccm_fit_rejects_unnormalized_rgb():
    preprocessor = ImagePreprocessor(calibration_method="linear")
    measured = np.tile([128, 64, 32], (24, 1))

    with pytest.raises(ValueError, match=r"normalized to \[0, 1\]"):
        preprocessor.compute_color_correction_matrix(
            measured, reference_id="before_nov_2014"
        )


def test_ccm_fit_rejects_empty_and_mismatched_patch_sets():
    preprocessor = ImagePreprocessor(calibration_method="linear")

    with pytest.raises(ValueError, match="must not be empty"):
        preprocessor.compute_color_correction_matrix(
            np.empty((0, 3)), np.empty((0, 3))
        )
    with pytest.raises(ValueError, match="same number"):
        preprocessor.compute_color_correction_matrix(
            np.ones((6, 3)) * 0.5, np.ones((7, 3)) * 50
        )


def test_ccm_fit_rejects_rank_deficient_patches():
    preprocessor = ImagePreprocessor(calibration_method="linear")
    measured = np.tile([0.2, 0.4, 0.6], (24, 1))

    with pytest.raises(ValueError, match="rank deficient"):
        preprocessor.compute_color_correction_matrix(
            measured, reference_id="before_nov_2014"
        )


def test_ccm_fit_accepts_integer_lab_after_float_conversion():
    rng = np.random.default_rng(7)
    measured = rng.uniform(0.05, 0.95, size=(24, 3))
    integer_lab = np.rint(COLORCHECKER_24_LAB_D50).astype(np.int64)
    preprocessor = ImagePreprocessor(calibration_method="linear")

    matrix = preprocessor.compute_color_correction_matrix(measured, integer_lab)

    assert matrix.shape == (3, 3)
    assert preprocessor.last_calibration_report is not None
    assert preprocessor.last_calibration_report.rank == 3

    report = preprocessor.compute_color_correction_matrix(
        measured, integer_lab, return_report=True
    )
    assert report.matrix.shape == (3, 3)
    assert set(report.training_delta_e00) == {"mean", "median", "p95", "max"}


@pytest.mark.parametrize("degree", [0, 4, 10])
def test_unsupported_polynomial_degree_is_rejected(degree):
    with pytest.raises(ValueError, match="polynomial_degree"):
        ImagePreprocessor(
            calibration_method="polynomial", polynomial_degree=degree
        )


def test_find_images_includes_uppercase_raf(tmp_path):
    raf_path = tmp_path / "leaf.RAF"
    raf_path.write_bytes(b"test")

    assert find_images(str(tmp_path)) == [raf_path]


def test_find_images_handles_mixed_case_extensions_in_one_recursive_walk(tmp_path):
    image_path = tmp_path / "nested" / "leaf.JpG"
    image_path.parent.mkdir()
    image_path.write_bytes(b"test")

    assert find_images(str(tmp_path)) == [image_path]


def test_raf_is_routed_through_raw_reader(tmp_path, monkeypatch):
    raf_path = tmp_path / "leaf.RAF"
    raf_path.write_bytes(b"test")
    expected = np.full((2, 3, 3), 0.5, dtype=np.float32)
    calls = []

    def fake_read_raw(path):
        calls.append(path)
        return expected.copy()

    preprocessor = ImagePreprocessor()
    monkeypatch.setattr(preprocessor, "read_raw", fake_read_raw)

    result = preprocessor.process(str(raf_path), white_balance_method="none")

    assert calls == [str(raf_path)]
    assert np.array_equal(result["rgb"], expected)


def test_raw_reader_uses_camera_wb_and_srgb_gamma_by_default(monkeypatch):
    captured = {}

    class FakeRaw:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

        def postprocess(self, **kwargs):
            captured.update(kwargs)
            return np.full((2, 3, 3), 32768, dtype=np.uint16)

    fake_rawpy = types.SimpleNamespace(
        imread=lambda path: FakeRaw(),
        ColorSpace=types.SimpleNamespace(sRGB="sRGB"),
    )
    monkeypatch.setitem(sys.modules, "rawpy", fake_rawpy)

    result = ImagePreprocessor.read_raw("leaf.RAF")

    assert captured["use_camera_wb"] is True
    assert captured["output_color"] == "sRGB"
    assert captured["gamma"] == (2.222, 4.5)
    assert captured["no_auto_bright"] is True
    assert captured["output_bps"] == 16
    assert np.allclose(result, 32768 / 65535)


def test_postprocess_removes_border_object_and_keeps_small_center_leaves():
    mask = np.zeros((200, 200), dtype=np.uint8)
    mask[0:180, 0:20] = 255
    mask[50:80, 70:100] = 255
    mask[110:135, 120:150] = 255

    segmenter = BaseSegmenter(
        morph_kernel_size=1,
        min_area_ratio=0.002,
        exclude_border_components=True,
        border_margin_ratio=0.01,
    )
    result = segmenter.postprocess(mask)

    assert not np.any(result[0:180, 0:20])
    assert np.all(result[50:80, 70:100] == 255)
    assert np.all(result[110:135, 120:150] == 255)


def test_grabcut_auto_seed_inherits_configured_area_and_border_filters(monkeypatch):
    captured = {}

    class FakeExGSegmenter:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def segment(self, image):
            mask = np.zeros(image.shape[:2], dtype=np.uint8)
            mask[3:7, 3:7] = 255
            return mask

    segmenter = GrabCutSegmenter(
        min_area_ratio=0.002,
        exclude_border_components=True,
        border_margin_ratio=0.03,
    )
    monkeypatch.setattr(segmentation_module, "ExGSegmenter", FakeExGSegmenter)
    monkeypatch.setattr(
        segmenter, "segment", lambda image, init_mask=None, **kwargs: init_mask
    )

    result = segmenter.segment_auto(np.zeros((10, 10, 3), dtype=np.float32))

    assert captured["min_area_ratio"] == 0.002
    assert captured["exclude_border_components"] is True
    assert captured["border_margin_ratio"] == 0.03
    assert np.any(result)


def test_grabcut_auto_keeps_proposal_when_refinement_loses_a_leaf(monkeypatch):
    proposal = np.zeros((30, 30), dtype=np.uint8)
    proposal[5:10, 5:10] = 255
    proposal[20:25, 20:25] = 255

    class FakeExGSegmenter:
        def __init__(self, **kwargs):
            pass

        def segment(self, image):
            return proposal.copy()

    refined = np.zeros_like(proposal)
    refined[20:25, 20:25] = 255
    segmenter = GrabCutSegmenter(min_area_ratio=0.002)
    monkeypatch.setattr(segmentation_module, "ExGSegmenter", FakeExGSegmenter)
    monkeypatch.setattr(
        segmenter, "segment", lambda image, init_mask=None, **kwargs: refined.copy()
    )

    result = segmenter.segment_auto(np.zeros((30, 30, 3), dtype=np.float32))

    assert np.array_equal(result, proposal)


def test_cli_override_defaults_are_none(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["batch_extract.py"])

    args = batch_extract.parse_args()

    assert Path(args.config) == batch_extract.DEFAULT_CONFIG_PATH
    assert args.method is None
    assert args.device is None
    assert args.white_balance is None


def test_config_values_survive_when_cli_does_not_override_them(tmp_path, monkeypatch):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
input:
  image_dir: images
  output_dir: output
imaging:
  white_balance: none
segmentation:
  method: auto
output:
  format: csv
""".strip(),
        encoding="utf-8",
    )
    captured = {}

    class FakePipeline:
        def __init__(self, config):
            captured["config"] = config
            self.last_batch_failures = []

        def process_batch(self, **kwargs):
            captured["kwargs"] = kwargs
            return pd.DataFrame({"GLI": [0.2]})

    monkeypatch.setattr(batch_extract, "LeafColorPipeline", FakePipeline)
    monkeypatch.setattr(sys, "argv", ["batch_extract.py", "--config", str(config_path)])

    assert batch_extract.main() == 0
    assert captured["config"]["segmentation"]["method"] == "auto"
    assert captured["kwargs"]["white_balance"] == "none"


@pytest.mark.parametrize(("filename", "expected"), [
    ("BJC-001_rep1_2024.jpg", "BJC-001"),
    ("BJC-001_2.jpg", "BJC-001"),
    ("sample_A1.jpg", "sample_A1"),
])
def test_default_sample_id_parsing_avoids_underscore_collisions(filename, expected):
    assert parse_sample_id(filename) == expected


def test_aggregate_preserves_metadata_and_distinguishes_rep_stats():
    frame = pd.DataFrame({
        "sample_id": ["A", "A", "B"],
        "image_path": ["a1.png", "a2.png", "b1.png"],
        "developmental_stage": ["heading", "heading", "seedling"],
        "treatment": ["control", "control", "salt"],
        "CIELAB_L_mean": [10.0, 20.0, 30.0],
        "CIELAB_L_std": [1.0, 3.0, 5.0],
    })

    result = LeafColorPipeline._aggregate_by_sample(
        frame, trait_columns=["CIELAB_L_mean", "CIELAB_L_std"]
    ).set_index("sample_id")

    assert result.loc["A", "CIELAB_L_mean"] == 15.0
    assert np.isclose(result.loc["A", "CIELAB_L_mean_rep_std"], np.sqrt(50.0))
    assert "CIELAB_L_std" in result.columns
    assert "CIELAB_L_std_rep_std" in result.columns
    assert result.loc["A", "n_replicates"] == 2
    assert result.loc["A", "developmental_stage"] == "heading"
    assert result.loc["A", "treatment"] == "control"


def test_aggregate_without_traits_still_counts_replicates():
    frame = pd.DataFrame({
        "sample_id": ["A", "A"],
        "developmental_stage": ["heading", "heading"],
    })

    result = LeafColorPipeline._aggregate_by_sample(
        frame, trait_columns=[]
    ).set_index("sample_id")

    assert result.loc["A", "n_replicates"] == 2
    assert result.loc["A", "developmental_stage"] == "heading"


def test_json_output_honors_explicit_extension(tmp_path):
    pipeline = LeafColorPipeline()
    frame = pd.DataFrame({"sample_id": ["A"], "GLI": [0.2]})

    path = pipeline._save_table(frame, str(tmp_path / "phenotypes.json"))

    assert path.suffix == ".json"
    assert json.loads(path.read_text(encoding="utf-8"))[0]["sample_id"] == "A"


def test_synthetic_image_runs_through_batch_pipeline(tmp_path):
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    image = np.full((64, 64, 3), 255, dtype=np.uint8)
    image[16:48, 20:44] = [30, 160, 30]
    write_image_rgb(image_dir / "plant-01_rep1.png", image)
    output_path = tmp_path / "phenotypes.csv"
    pipeline = LeafColorPipeline({
        "segmentation": {"method": "exg", "min_leaf_area_ratio": 0.01},
        "features": {"texture": {"enabled": False}},
    })

    result = pipeline.process_batch(
        str(image_dir), output_csv=str(output_path), group_by_sample=False,
        white_balance="none", verbose=False,
    )

    assert len(result) == 1
    assert result.loc[0, "QC_mask_area_px"] > 0
    assert output_path.exists()
    manifest = json.loads(
        (tmp_path / "phenotypes_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["successful_images"] == 1
    assert manifest["failed_images"] == 0
    assert len(manifest["config_sha256"]) == 64
    assert pipeline.last_batch_failures == []


def test_ccm_changes_color_features_without_changing_segmentation_mask(tmp_path):
    image = np.full((80, 80, 3), 255, dtype=np.uint8)
    image[20:60, 25:55] = [30, 160, 30]
    image_path = tmp_path / "leaf.png"
    write_image_rgb(image_path, image)
    common = {
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
    uncalibrated = LeafColorPipeline(common)
    calibrated = LeafColorPipeline({
        **common,
        "color_calibration": {
            "enabled": True,
            "method": "linear",
            "matrix": [[0, 1, 0], [1, 0, 0], [0, 0, 1]],
        },
    })

    baseline = uncalibrated.process_single(
        str(image_path), return_visualization=True, verbose=False
    )
    corrected = calibrated.process_single(
        str(image_path), return_visualization=True, verbose=False
    )

    assert np.array_equal(baseline["mask"], corrected["mask"])
    assert baseline["features"]["RGB_G_mean"] > baseline["features"]["RGB_R_mean"]
    assert corrected["features"]["RGB_R_mean"] > corrected["features"]["RGB_G_mean"]


def test_pipeline_applies_ccm_only_after_leaf_crop(tmp_path, monkeypatch):
    image = np.full((100, 120, 3), 255, dtype=np.uint8)
    image[30:70, 45:75] = [30, 160, 30]
    image_path = tmp_path / "leaf.png"
    write_image_rgb(image_path, image)
    pipeline = LeafColorPipeline({
        "color_calibration": {
            "enabled": True,
            "method": "linear",
            "matrix": np.eye(3).tolist(),
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
    })
    seen_shapes = []
    apply_ccm = pipeline.preprocessor.apply_color_correction

    def record_shape(rgb):
        seen_shapes.append(rgb.shape)
        return apply_ccm(rgb)

    monkeypatch.setattr(pipeline.preprocessor, "apply_color_correction", record_shape)

    pipeline.process_single(str(image_path), verbose=False)

    assert len(seen_shapes) == 1
    assert seen_shapes[0][0] < image.shape[0]
    assert seen_shapes[0][1] < image.shape[1]


def test_process_single_rejects_metadata_that_overwrites_reserved_fields(tmp_path):
    image = np.full((32, 32, 3), 255, dtype=np.uint8)
    image[8:24, 8:24] = [30, 160, 30]
    image_path = tmp_path / "leaf.png"
    write_image_rgb(image_path, image)
    pipeline = LeafColorPipeline({
        "segmentation": {"method": "exg", "min_leaf_area_ratio": 0.01},
        "features": {
            "vegetation_indices": {"enabled": False},
            "texture": {"enabled": False},
            "shape": {"enabled": False},
        },
    })

    with pytest.raises(ValueError, match="metadata keys collide.*sample_id"):
        pipeline.process_single(
            str(image_path),
            metadata={"sample_id": "wrong"},
            verbose=False,
        )


def test_all_batch_failures_are_written_to_a_report(tmp_path, monkeypatch):
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    (image_dir / "broken.jpg").write_bytes(b"not an image")
    output_path = tmp_path / "phenotypes.csv"
    pipeline = LeafColorPipeline()

    def fail_processing(*args, **kwargs):
        raise ValueError("synthetic failure")

    monkeypatch.setattr(pipeline, "process_single", fail_processing)
    result = pipeline.process_batch(
        str(image_dir), output_csv=str(output_path), verbose=False,
    )

    failure_path = tmp_path / "phenotypes_failures.csv"
    assert result.empty
    assert len(pipeline.last_batch_failures) == 1
    assert failure_path.exists()
    assert "synthetic failure" in failure_path.read_text(encoding="utf-8")


def test_clean_rerun_removes_stale_failure_report(tmp_path):
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    image = np.full((32, 32, 3), 255, dtype=np.uint8)
    image[8:24, 8:24] = [30, 160, 30]
    write_image_rgb(image_dir / "leaf.png", image)
    broken = image_dir / "broken.png"
    broken.write_bytes(b"not an image")
    output_path = tmp_path / "phenotypes.csv"
    config = {
        "segmentation": {"method": "exg", "min_leaf_area_ratio": 0.01},
        "features": {"texture": {"enabled": False}},
    }

    first = LeafColorPipeline(config)
    first.process_batch(
        str(image_dir), output_csv=str(output_path),
        group_by_sample=False, verbose=False,
    )
    failure_path = tmp_path / "phenotypes_failures.csv"
    assert failure_path.is_file()

    broken.unlink()
    second = LeafColorPipeline(config)
    second.process_batch(
        str(image_dir), output_csv=str(output_path),
        group_by_sample=False, verbose=False,
    )

    assert second.last_batch_failures == []
    assert not failure_path.exists()


def test_nonaggregated_rerun_removes_stale_raw_table(tmp_path):
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    image = np.full((32, 32, 3), 255, dtype=np.uint8)
    image[8:24, 8:24] = [30, 160, 30]
    write_image_rgb(image_dir / "leaf_rep1.png", image)
    write_image_rgb(image_dir / "leaf_rep2.png", image)
    output_path = tmp_path / "phenotypes.csv"
    config = {
        "segmentation": {"method": "exg", "min_leaf_area_ratio": 0.01},
        "features": {"texture": {"enabled": False}},
    }

    LeafColorPipeline(config).process_batch(
        str(image_dir), output_csv=str(output_path),
        group_by_sample=True, verbose=False,
    )
    raw_path = tmp_path / "phenotypes_raw.csv"
    assert raw_path.is_file()

    LeafColorPipeline(config).process_batch(
        str(image_dir), output_csv=str(output_path),
        group_by_sample=False, verbose=False,
    )

    assert not raw_path.exists()


def test_sixteen_bit_standard_image_preserves_dynamic_range(tmp_path):
    image = np.array([[[0, 32768, 65535]]], dtype=np.uint16)
    bgr = image[..., ::-1]
    ok, encoded = segmentation_module.cv2.imencode(".png", bgr)
    assert ok
    path = tmp_path / "sixteen_bit.png"
    encoded.tofile(str(path))

    result = read_image_rgb(path, as_float=True)

    assert result.dtype == np.float32
    assert np.allclose(result[0, 0], [0.0, 32768 / 65535, 1.0])


def test_largest_component_policy_is_reflected_in_qc():
    raw_mask = np.zeros((20, 20), dtype=np.uint8)
    raw_mask[2:12, 2:12] = 255
    raw_mask[15:20, 15:20] = 255

    selected = LeafColorPipeline._select_analysis_mask(raw_mask, "largest")
    qc = LeafColorPipeline._mask_qc(selected, raw_mask=raw_mask)

    assert np.count_nonzero(selected) == 100
    assert qc["QC_component_count"] == 2
    assert qc["QC_raw_mask_area_px"] == 125
    assert np.isclose(qc["QC_largest_component_fraction"], 0.8)


def test_component_min_exg_rejects_nonvegetation_instead_of_falling_back():
    gray_image = np.full((20, 20, 3), 0.5, dtype=np.float32)
    single_component = np.zeros((20, 20), dtype=np.uint8)
    single_component[2:12, 2:12] = 255

    selected = LeafColorPipeline._select_analysis_mask(
        single_component,
        "largest",
        img_rgb=gray_image,
        min_exg=0.30,
    )

    assert not np.any(selected)


def test_component_min_exg_selects_smaller_valid_vegetation_component():
    image = np.full((30, 30, 3), 0.5, dtype=np.float32)
    raw_mask = np.zeros((30, 30), dtype=np.uint8)
    raw_mask[2:16, 2:16] = 255
    raw_mask[20:26, 20:26] = 255
    image[20:26, 20:26] = [0.1, 0.8, 0.1]

    selected = LeafColorPipeline._select_analysis_mask(
        raw_mask,
        "largest",
        img_rgb=image,
        min_exg=0.30,
    )

    assert np.count_nonzero(selected) == 36
    assert np.all(selected[20:26, 20:26] == 255)
    assert not np.any(selected[2:16, 2:16])


def test_recursive_visualization_paths_do_not_collide(tmp_path):
    image_root = tmp_path / "images"
    vis_root = tmp_path / "visualizations"
    first = image_root / "batch-a" / "leaf.jpg"
    second = image_root / "batch-b" / "leaf.jpg"
    third = image_root / "batch-a" / "leaf.png"

    paths = {
        LeafColorPipeline._visualization_output_path(vis_root, image_root, path)
        for path in (first, second, third)
    }

    assert len(paths) == 3
    assert vis_root / "batch-a" / "leaf__jpg_vis.png" in paths
    assert vis_root / "batch-b" / "leaf__jpg_vis.png" in paths
    assert vis_root / "batch-a" / "leaf__png_vis.png" in paths


def test_batch_writes_both_recursive_visualizations_with_duplicate_stems(tmp_path):
    image_root = tmp_path / "images"
    first = image_root / "batch-a" / "leaf.png"
    second = image_root / "batch-b" / "leaf.png"
    image = np.full((32, 32, 3), 255, dtype=np.uint8)
    image[8:24, 8:24] = [30, 160, 30]
    write_image_rgb(first, image)
    write_image_rgb(second, image)
    vis_root = tmp_path / "visualizations"
    pipeline = LeafColorPipeline({
        "segmentation": {"method": "exg", "min_leaf_area_ratio": 0.01},
        "features": {"texture": {"enabled": False}},
    })

    result = pipeline.process_batch(
        str(image_root),
        group_by_sample=False,
        save_visualizations=True,
        visualization_dir=str(vis_root),
        verbose=False,
    )

    assert len(result) == 2
    assert (vis_root / "batch-a" / "leaf__png_vis.png").exists()
    assert (vis_root / "batch-b" / "leaf__png_vis.png").exists()


def test_batch_does_not_reingest_visualizations_inside_input_tree(tmp_path):
    image_root = tmp_path / "images"
    image = np.full((32, 32, 3), 255, dtype=np.uint8)
    image[8:24, 8:24] = [30, 160, 30]
    write_image_rgb(image_root / "leaf.png", image)
    output_path = image_root / "phenotypes.csv"
    config = {
        "segmentation": {"method": "exg", "min_leaf_area_ratio": 0.01},
        "features": {"texture": {"enabled": False}},
    }

    LeafColorPipeline(config).process_batch(
        str(image_root), output_csv=str(output_path),
        group_by_sample=False, save_visualizations=True, verbose=False,
    )
    result = LeafColorPipeline(config).process_batch(
        str(image_root), output_csv=str(output_path),
        group_by_sample=False, save_visualizations=True, verbose=False,
    )

    assert len(result) == 1
    assert Path(result.iloc[0]["image_path"]).name == "leaf.png"


def test_aggregate_rejects_conflicting_metadata_within_sample():
    frame = pd.DataFrame({
        "sample_id": ["A", "A"],
        "image_path": ["a1.png", "a2.png"],
        "treatment": ["control", "salt"],
        "GLI": [0.2, 0.4],
    })

    with pytest.raises(ValueError, match="Conflicting metadata.*treatment"):
        LeafColorPipeline._aggregate_by_sample(frame, trait_columns=["GLI"])


def test_aggregate_without_traits_also_rejects_conflicting_metadata():
    frame = pd.DataFrame({
        "sample_id": ["A", "A"],
        "developmental_stage": ["seedling", "heading"],
    })

    with pytest.raises(ValueError, match="Conflicting metadata.*developmental_stage"):
        LeafColorPipeline._aggregate_by_sample(frame, trait_columns=[])


def test_white_tissue_filter_removes_low_saturation_pixels_inside_leaf():
    mask = np.full((10, 10), 255, dtype=np.uint8)
    image = np.full((10, 10, 3), [0.1, 0.7, 0.1], dtype=np.float32)
    image[2:4, 2:7] = [0.8, 0.8, 0.8]

    refined, qc = LeafColorPipeline._exclude_white_tissue(
        mask,
        image,
        max_saturation=0.25,
        min_retained_fraction=0.50,
    )

    assert np.count_nonzero(refined) == 90
    assert np.all(refined[2:4, 2:7] == 0)
    assert np.all(refined[0:2] == 255)
    assert qc["QC_white_tissue_removed_px"] == 10
    assert np.isclose(qc["QC_white_tissue_removed_fraction"], 0.10)
    assert qc["QC_white_tissue_filter_rollback"] == 0

    mask_qc = LeafColorPipeline._mask_qc(
        refined,
        raw_mask=mask,
        selected_component_mask=mask,
    )
    assert np.isclose(mask_qc["QC_mask_area_ratio"], 0.90)
    assert np.isclose(mask_qc["QC_selected_component_fraction"], 1.0)


def test_white_tissue_filter_rolls_back_when_too_little_leaf_would_remain():
    mask = np.full((10, 10), 255, dtype=np.uint8)
    image = np.full((10, 10, 3), 0.8, dtype=np.float32)
    image[0, 0] = [0.1, 0.7, 0.1]

    refined, qc = LeafColorPipeline._exclude_white_tissue(
        mask,
        image,
        max_saturation=0.25,
        min_retained_fraction=0.50,
    )

    assert np.array_equal(refined, mask)
    assert qc["QC_white_tissue_removed_px"] == 0
    assert qc["QC_white_tissue_removed_fraction"] == 0
    assert qc["QC_white_tissue_filter_rollback"] == 1


def test_white_tissue_filter_config_is_validated():
    with pytest.raises(ValueError, match="white_tissue_max_saturation"):
        LeafColorPipeline({
            "segmentation": {"white_tissue_max_saturation": 1.01}
        })
    with pytest.raises(ValueError, match="white_tissue_min_retained_fraction"):
        LeafColorPipeline({
            "segmentation": {"white_tissue_min_retained_fraction": 0}
        })


def test_single_replicates_do_not_create_all_nan_stat_columns():
    frame = pd.DataFrame({
        "sample_id": ["A", "B"],
        "image_path": ["a.png", "b.png"],
        "GLI": [0.2, 0.3],
    })

    result = LeafColorPipeline._aggregate_by_sample(frame, trait_columns=["GLI"])

    assert "GLI_rep_std" not in result.columns
    assert "GLI_rep_cv" not in result.columns
    assert result["n_replicates"].tolist() == [1, 1]


def test_unknown_segmentation_parameter_is_rejected():
    with pytest.raises(ValueError, match="Unknown segmentation parameters"):
        LeafColorPipeline({"segmentation": {"method": "auto", "morph_kernal_size": 5}})


def test_segmentation_uses_bounded_proxy_and_restores_mask_size():
    pipeline = LeafColorPipeline({
        "segmentation": {"method": "exg", "max_processing_dimension": 256}
    })
    seen_shapes = []

    class RecordingSegmenter:
        def segment(self, image):
            seen_shapes.append(image.shape)
            return np.full(image.shape[:2], 255, dtype=np.uint8)

    pipeline.segmenter = RecordingSegmenter()
    image = np.zeros((400, 800, 3), dtype=np.float32)

    mask = pipeline._segment_image(image)

    assert seen_shapes == [(128, 256, 3)]
    assert mask.shape == (400, 800)


def test_unet_checkpoint_metadata_is_loaded_with_restricted_deserialization(monkeypatch):
    captured = {}

    class FakeModel:
        def load_state_dict(self, state_dict):
            captured["state_dict"] = state_dict

        def to(self, device):
            captured["model_device"] = device
            return self

        def eval(self):
            captured["evaluated"] = True
            return self

    def fake_load(path, map_location=None, weights_only=False):
        captured["load"] = {
            "path": path,
            "map_location": map_location,
            "weights_only": weights_only,
        }
        return {
            "state_dict": {"weight": "sentinel"},
            "backbone": "resnet34",
            "image_size": [320, 480],
            "normalization": {
                "mean": [0.1, 0.2, 0.3],
                "std": [0.4, 0.5, 0.6],
            },
            "threshold": 0.65,
        }

    fake_torch = types.ModuleType("torch")
    fake_torch.cuda = types.SimpleNamespace(is_available=lambda: False)
    fake_torch.load = fake_load
    fake_smp = types.ModuleType("segmentation_models_pytorch")
    fake_smp.Unet = lambda **kwargs: FakeModel()
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "segmentation_models_pytorch", fake_smp)

    segmenter = UNetSegmenter(
        "checkpoint.pth", backbone="resnet34", device="cpu"
    )
    segmenter._load_model()

    assert captured["load"]["weights_only"] is True
    assert captured["state_dict"] == {"weight": "sentinel"}
    assert segmenter.input_size == (320, 480)
    assert np.allclose(segmenter.normalization_mean, [0.1, 0.2, 0.3])
    assert np.allclose(segmenter.normalization_std, [0.4, 0.5, 0.6])
    assert segmenter.threshold == 0.65


def test_training_split_keeps_sample_replicates_together():
    pairs = [
        (Path("A_rep1.jpg"), Path("A_rep1.png")),
        (Path("A_rep2.jpg"), Path("A_rep2.png")),
        (Path("B_rep1.jpg"), Path("B_rep1.png")),
        (Path("B_rep2.jpg"), Path("B_rep2.png")),
        (Path("C_rep1.jpg"), Path("C_rep1.png")),
    ]

    train_pairs, val_pairs = split_pairs_by_sample(pairs, seed=7)
    train_ids = {parse_sample_id(pair[0].name) for pair in train_pairs}
    val_ids = {parse_sample_id(pair[0].name) for pair in val_pairs}

    assert train_ids.isdisjoint(val_ids)
    assert train_ids | val_ids == {"A", "B", "C"}
