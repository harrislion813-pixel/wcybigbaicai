import numpy as np

from src.color_features import ColorFeatureExtractor
from src.preprocessing import ImagePreprocessor
from src.segmentation import IMAGENET_MEAN, IMAGENET_STD, normalize_imagenet_rgb
from src.texture_features import ColorTextureAnalyzer, GLCMTextureExtractor
from src.utils import (
    COLORCHECKER_24_LAB_D50,
    delta_e_2000,
    get_colorchecker_lab_d65,
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


def test_colorchecker_d50_white_adapts_to_neutral_srgb():
    white_lab_d50 = np.array([[100.0, 0.0, 0.0]], dtype=np.float64)

    srgb = ImagePreprocessor._lab_to_srgb_approx(white_lab_d50)[0]

    assert np.allclose(srgb, [1.0, 1.0, 1.0], atol=2e-4)


def test_colorchecker_d65_reference_is_actually_adapted():
    adapted = get_colorchecker_lab_d65()

    assert adapted.shape == (24, 3)
    assert np.isfinite(adapted).all()
    assert not np.allclose(adapted, COLORCHECKER_24_LAB_D50, atol=1e-6)


def test_pixel_channel_ratios_ignore_near_black_denominators():
    image = np.array([[[255, 0, 255], [100, 100, 50]]], dtype=np.uint8)
    mask = np.full((1, 2), 255, dtype=np.uint8)
    extractor = ColorFeatureExtractor(
        color_spaces=["RGB"],
        include_color_moments=False,
        include_histogram=False,
        include_chromaticity=False,
    )

    result = extractor.extract(image, mask)

    assert result["RGB_ratio_valid_fraction"] == 0.5
    assert np.isclose(result["RGB_B_over_G_std"], 0.0)
    assert np.isfinite(result["RGB_B_over_G_mean"])


def test_hue_statistics_are_circular_at_opencv_wraparound():
    hsv = np.array([[[1, 255, 255], [179, 255, 255]]], dtype=np.uint8)
    mask = np.ones((1, 2), dtype=np.uint8)
    extractor = ColorFeatureExtractor(color_spaces=[])

    result = extractor._extract_space_features(hsv, mask, "HSV", ["H", "S", "V"])

    assert result["HSV_H_mean"] < 2 or result["HSV_H_mean"] > 178
    assert result["HSV_H_std"] < 2


def test_masked_glcm_ignores_pixels_outside_leaf_mask():
    image_a = np.zeros((8, 8, 3), dtype=np.uint8)
    image_b = np.full((8, 8, 3), 255, dtype=np.uint8)
    leaf_pattern = np.tile(np.array([40, 80, 40, 80], dtype=np.uint8), (4, 1))
    image_a[2:6, 2:6] = leaf_pattern[..., None]
    image_b[2:6, 2:6] = leaf_pattern[..., None]
    mask = np.zeros((8, 8), dtype=np.uint8)
    mask[2:6, 2:6] = 255
    extractor = GLCMTextureExtractor(distances=[1], angles=[0], levels=16)

    result_a = extractor.compute(image_a, mask)
    result_b = extractor.compute(image_b, mask)

    assert result_a == result_b


def test_signed_lab_uniformity_cv_is_non_negative():
    lab = np.array([[[-20.0, -2.0, 3.0], [-10.0, -4.0, 5.0]]], dtype=np.float32)
    mask = np.full((1, 2), 255, dtype=np.uint8)

    result = ColorTextureAnalyzer.color_uniformity(lab, mask)

    assert result["Uniformity_CV_a"] >= 0
    assert result["Uniformity_CV_b"] >= 0


def test_feature_names_have_no_case_insensitive_collisions():
    image = np.full((4, 4, 3), [0.2, 0.6, 0.1], dtype=np.float32)
    mask = np.full((4, 4), 255, dtype=np.uint8)

    result = ColorFeatureExtractor().extract(image, mask)
    normalized_names = [name.casefold() for name in result]

    assert len(normalized_names) == len(set(normalized_names))

