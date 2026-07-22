import json
import sys

import numpy as np
import pandas as pd
import pytest

from scripts import batch_extract
from src.pipeline import LeafColorPipeline
from src.preprocessing import ImagePreprocessor
from src.utils import parse_sample_id, write_image_rgb


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


def test_cli_override_defaults_are_none(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["batch_extract.py"])

    args = batch_extract.parse_args()

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
