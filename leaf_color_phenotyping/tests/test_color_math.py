import numpy as np

from src.color_features import ColorFeatureExtractor
from src.segmentation import IMAGENET_MEAN, IMAGENET_STD, normalize_imagenet_rgb
from src.utils import (
    delta_e_2000,
    histogram_features,
    rgb_to_lab,
    rgb_to_ycbcr,
)
from src.vegetation_indices import VegetationIndexExtractor


def test_histogram_bins_use_fixed_uint8_range():
    low = histogram_features(np.array([0, 1], dtype=np.uint8), bins=2)
    high = histogram_features(np.array([254, 255], dtype=np.uint8), bins=2)

    assert low == {"hist_bin_0": 1.0, "hist_bin_1": 0.0}
    assert high == {"hist_bin_0": 0.0, "hist_bin_1": 1.0}


def test_ycbcr_channels_are_returned_as_y_cb_cr():
    red = np.array([[[1.0, 0.0, 0.0]]], dtype=np.float32)
    y, cb, cr = rgb_to_ycbcr(red)[0, 0]

    assert y == 76
    assert cb == 85
    assert cr == 255


def test_cive_uses_eight_bit_equivalent_rgb_scale():
    rgb = np.array([[[0.3, 0.5, 0.2]]], dtype=np.float32)
    mask = np.array([[255]], dtype=np.uint8)
    extractor = VegetationIndexExtractor(indices=["CIVE"])

    result = extractor.compute(rgb, mask)
    expected = 0.441 * 76.5 - 0.811 * 127.5 + 0.385 * 51.0 + 18.78745

    assert np.isclose(result["CIVE"], expected, atol=1e-5)


def test_srgb_primary_red_matches_reference_lab():
    red = np.array([[[1.0, 0.0, 0.0]]], dtype=np.float32)
    lab = rgb_to_lab(red)[0, 0]

    assert np.allclose(lab, [53.2408, 80.0925, 67.2032], atol=1e-3)


def test_ciede2000_matches_published_reference_pair():
    lab1 = np.array([50.0, 2.6772, -79.7751])
    lab2 = np.array([50.0, 0.0, -82.7485])

    assert np.isclose(delta_e_2000(lab1, lab2), 2.0425, atol=1e-4)


def test_imagenet_normalization_is_shared_and_deterministic():
    black = np.zeros((1, 1, 3), dtype=np.float32)
    normalized = normalize_imagenet_rgb(black)[0, 0]

    assert np.allclose(normalized, -IMAGENET_MEAN / IMAGENET_STD)


def test_color_extractor_respects_explicit_empty_color_spaces():
    extractor = ColorFeatureExtractor(
        color_spaces=[],
        include_color_moments=False,
        include_histogram=False,
        include_chromaticity=False,
    )
    image = np.zeros((2, 2, 3), dtype=np.uint8)
    mask = np.full((2, 2), 255, dtype=np.uint8)

    assert extractor.extract(image, mask) == {}

