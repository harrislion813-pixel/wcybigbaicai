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
from src.segmentation import BaseSegmenter, GrabCutSegmenter
from src.utils import (
    find_images, parse_sample_id, read_image_rgb, split_pairs_by_sample,
    write_image_rgb,
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


def test_find_images_includes_uppercase_raf(tmp_path):
    raf_path = tmp_path / "leaf.RAF"
    raf_path.write_bytes(b"test")

    assert find_images(str(tmp_path)) == [raf_path]


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
