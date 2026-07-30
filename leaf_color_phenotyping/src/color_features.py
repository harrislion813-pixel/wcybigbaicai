"""
多颜色空间特征提取 — 这是整个项目的核心模块。

从叶片ROI中提取以下维度的颜色特征:
    1. RGB 空间: 通道均值/比值/归一化值
    2. HSV 空间: 色相/饱和度/明度统计
    3. CIELAB 空间: L*/a*/b* 统计 + ΔE参考
    4. YCbCr 空间: 亮度/色度分量
    5. 颜色矩: 每通道均值/标准差/偏度/峰度
    6. 直方图特征: 归一化直方图 + 关键百分位
    7. 色度坐标: CIE xyY, CIE u'v'
    8. 比值特征: R/G, B/G, (R-G)/(R+G) 等
"""
from typing import Dict, List, Optional, Tuple
import numpy as np
import cv2

from .utils import (
    rgb_to_lab, rgb_to_hsv, rgb_to_ycbcr,
    rgb_to_xyz, channel_stats, histogram_features,
)


class ColorFeatureExtractor:
    """多颜色空间叶色特征提取器.

    Usage:
        extractor = ColorFeatureExtractor(config)
        features = extractor.extract(img_rgb, mask)
        # features 是一个扁平 dict, 可直接转成 DataFrame 行
    """

    def __init__(self,
                 color_spaces: List[str] = None,
                 hist_bins: int = 32,
                 hist_percentiles: List[int] = None,
                 include_color_moments: bool = True,
                 include_histogram: bool = True,
                 include_chromaticity: bool = True):
        """
        Args:
            color_spaces: 要提取的颜色空间列表
            hist_bins: 直方图bin数
            hist_percentiles: 直方图百分位列表
            include_color_moments: 是否提取颜色矩
            include_histogram: 是否提取直方图特征
            include_chromaticity: 是否提取色度坐标特征
        """
        self.color_spaces = (["RGB", "HSV", "CIELAB", "YCbCr"]
                             if color_spaces is None else list(color_spaces))
        self.hist_bins = hist_bins
        self.hist_percentiles = ([10, 25, 50, 75, 90]
                                 if hist_percentiles is None else list(hist_percentiles))
        self.include_color_moments = include_color_moments
        self.include_histogram = include_histogram
        self.include_chromaticity = include_chromaticity
        valid_spaces = {"RGB", "HSV", "CIELAB", "YCbCr"}
        unknown_spaces = sorted(set(self.color_spaces) - valid_spaces)
        if unknown_spaces:
            raise ValueError(f"Unknown color spaces: {unknown_spaces}")
        if not isinstance(self.hist_bins, int) or self.hist_bins <= 0:
            raise ValueError("hist_bins must be a positive integer")
        if any(not 0 <= percentile <= 100 for percentile in self.hist_percentiles):
            raise ValueError("hist_percentiles must be within [0, 100]")

    def extract(self, img_rgb: np.ndarray,
                mask: np.ndarray,
                precomputed: Optional[Dict[str, np.ndarray]] = None
                ) -> Dict[str, float]:
        """从叶片ROI提取所有颜色特征.

        Args:
            img_rgb: RGB图像, (H,W,3), uint8 [0,255] or float32 [0,1]
            mask: 叶片ROI掩膜, (H,W), uint8, 255=叶片, 0=背景

        Returns:
            扁平特征字典, key格式: "{color_space}_{channel}_{statistic}"
        """
        # 输入规范化
        if img_rgb.dtype != np.uint8:
            img_uint8 = (img_rgb * 255).clip(0, 255).astype(np.uint8)
        else:
            img_uint8 = img_rgb

        if img_rgb.dtype == np.uint8 and img_rgb.max() <= 1.0:
            # 实际是 [0,1] 但被存为 uint8 了
            img_float = img_rgb.astype(np.float32)
        elif img_rgb.max() > 1.0:
            img_float = img_rgb.astype(np.float32) / 255.0
        else:
            img_float = img_rgb.astype(np.float32)

        # 掩膜规范化
        if mask.max() > 1:
            mask_bin = (mask > 127).astype(np.uint8)
        else:
            mask_bin = mask.astype(np.uint8)
        if not np.any(mask_bin):
            return {}

        precomputed = precomputed or {}
        all_features: Dict[str, float] = {}

        # ---- 1. RGB 空间 ----
        if "RGB" in self.color_spaces:
            all_features.update(self._extract_rgb_features(img_float, mask_bin))

        # ---- 2. HSV 空间 ----
        if "HSV" in self.color_spaces:
            hsv = precomputed.get("HSV")
            if hsv is None:
                hsv = rgb_to_hsv(img_float)
            all_features.update(self._extract_space_features(hsv, mask_bin, "HSV",
                                                             ch_names=["H", "S", "V"]))

        # ---- 3. CIELAB 空间 ----
        if "CIELAB" in self.color_spaces:
            lab = precomputed.get("CIELAB")
            if lab is None:
                lab = rgb_to_lab(img_float)
            all_features.update(self._extract_space_features(lab, mask_bin, "CIELAB",
                                                             ch_names=["L", "A", "B"]))
            # 额外: 色度 C*ab, 色相角 hab
            all_features.update(self._extract_lab_advanced(lab, mask_bin))

        # ---- 4. YCbCr 空间 ----
        if "YCbCr" in self.color_spaces:
            ycbcr = rgb_to_ycbcr(img_float)
            all_features.update(self._extract_space_features(ycbcr, mask_bin, "YCbCr",
                                                             ch_names=["Y", "Cb", "Cr"]))

        # ---- 5. 颜色矩 ----
        if self.include_color_moments:
            all_features.update(self._extract_color_moments(img_float, mask_bin))

        # ---- 6. 直方图 (仅在RGB上做) ----
        if self.include_histogram:
            all_features.update(self._extract_histogram_features(img_uint8, mask_bin))

        # ---- 7. 色度坐标 ----
        if self.include_chromaticity:
            all_features.update(self._extract_chromaticity_features(img_float, mask_bin))

        return all_features

    # ----------------------------------------------------------
    # RGB 专项特征
    # ----------------------------------------------------------
    def _extract_rgb_features(self, img: np.ndarray,
                              mask: np.ndarray) -> Dict[str, float]:
        """提取RGB空间所有特征."""
        feats = {}

        r_ch = img[..., 0][mask > 0]
        g_ch = img[..., 1][mask > 0]
        b_ch = img[..., 2][mask > 0]

        if len(r_ch) == 0:
            return feats

        r, g, b = float(r_ch.mean()), float(g_ch.mean()), float(b_ch.mean())

        # 通道均值 (归一化)
        feats["RGB_R_mean"] = r
        feats["RGB_G_mean"] = g
        feats["RGB_B_mean"] = b

        # 归一化通道值 (光照不变)
        total = r + g + b + 1e-10
        feats["RGB_r_norm"] = r / total
        feats["RGB_g_norm"] = g / total
        feats["RGB_b_norm"] = b / total

        # 通道比值
        feats["RGB_R_over_G"] = r / (g + 1e-10)
        feats["RGB_B_over_G"] = b / (g + 1e-10)
        feats["RGB_R_over_B"] = r / (b + 1e-10)

        # 归一化差值
        feats["RGB_NGRDI"] = (g - r) / (g + r + 1e-10)  # Normalized Green-Red
        feats["RGB_RG_ratio"] = (r - g) / (r + g + 1e-10)
        feats["RGB_BG_ratio"] = (b - g) / (b + g + 1e-10)
        feats["RGB_RB_ratio"] = (r - b) / (r + b + 1e-10)

        # 逐像素比值统计 (生物学上更有意义)
        # Very dark green-channel pixels make ratios numerically unbounded.
        # Exclude pixels below two 8-bit code values and report the retained share.
        ratio_floor = 2.0 / 255.0
        ratio_valid = g_ch > ratio_floor
        feats["RGB_ratio_valid_fraction"] = float(ratio_valid.mean())
        if np.any(ratio_valid):
            pixel_ratios_rg = r_ch[ratio_valid] / g_ch[ratio_valid]
            pixel_ratios_bg = b_ch[ratio_valid] / g_ch[ratio_valid]
            feats["RGB_R_over_G_mean"] = float(pixel_ratios_rg.mean())
            feats["RGB_R_over_G_std"] = float(pixel_ratios_rg.std())
            feats["RGB_B_over_G_mean"] = float(pixel_ratios_bg.mean())
            feats["RGB_B_over_G_std"] = float(pixel_ratios_bg.std())
        else:
            for name in (
                "RGB_R_over_G_mean", "RGB_R_over_G_std",
                "RGB_B_over_G_mean", "RGB_B_over_G_std",
            ):
                feats[name] = np.nan

        return feats

    # ----------------------------------------------------------
    # 通用颜色空间特征提取
    # ----------------------------------------------------------
    def _extract_space_features(self, img: np.ndarray, mask: np.ndarray,
                                space_name: str,
                                ch_names: List[str]) -> Dict[str, float]:
        """提取任意颜色空间的基础特征."""
        feats = {}
        for i, name in enumerate(ch_names):
            ch = img[..., i][mask > 0]
            if len(ch) == 0:
                continue

            # 归一化 (若输入为uint8,需要做)
            if ch.dtype == np.uint8:
                ch = ch.astype(np.float32)

            stats = channel_stats(ch, self.hist_percentiles)
            if space_name == "HSV" and name == "H":
                # OpenCV hue is circular on [0, 180); replace linear mean/std.
                angles = ch.astype(np.float64) * (2 * np.pi / 180.0)
                sin_mean = np.sin(angles).mean()
                cos_mean = np.cos(angles).mean()
                mean_angle = np.arctan2(sin_mean, cos_mean) % (2 * np.pi)
                resultant = float(np.hypot(sin_mean, cos_mean))
                stats["mean"] = float(mean_angle * 180.0 / (2 * np.pi))
                stats["std"] = float(
                    np.sqrt(max(0.0, -2.0 * np.log(max(resultant, 1e-12))))
                    * 180.0 / (2 * np.pi)
                )
                stats["circular_variance"] = 1.0 - resultant
            for stat_name, value in stats.items():
                feats[f"{space_name}_{name}_{stat_name}"] = float(value)

        return feats

    # ----------------------------------------------------------
    # CIELAB 高级特征
    # ----------------------------------------------------------
    def _extract_lab_advanced(self, lab: np.ndarray,
                              mask: np.ndarray) -> Dict[str, float]:
        """提取CIELAB高级特征: 色度C*ab, 色相角hab, ΔE."""
        feats = {}
        a_ch = lab[..., 1][mask > 0]
        b_ch = lab[..., 2][mask > 0]
        L_ch = lab[..., 0][mask > 0]

        if len(a_ch) == 0:
            return feats

        # 色度 C*ab = sqrt(a² + b²)
        C_ab = np.sqrt(a_ch**2 + b_ch**2)
        feats["CIELAB_Cab_mean"] = float(C_ab.mean())
        feats["CIELAB_Cab_std"] = float(C_ab.std())
        feats["CIELAB_Cab_median"] = float(np.median(C_ab))

        # 色相角 hab = arctan(b/a) (度)
        h_ab = np.arctan2(b_ch, a_ch + 1e-10) * 180 / np.pi
        h_ab = np.where(h_ab < 0, h_ab + 360, h_ab)
        # 圆形统计: 用向量平均法
        h_rad = np.deg2rad(h_ab)
        sin_mean = float(np.sin(h_rad).mean())
        cos_mean = float(np.cos(h_rad).mean())
        h_ab_mean = float(np.arctan2(sin_mean, cos_mean) * 180 / np.pi)
        if h_ab_mean < 0:
            h_ab_mean += 360
        feats["CIELAB_hab_mean"] = h_ab_mean
        resultant = float(np.hypot(sin_mean, cos_mean))
        feats["CIELAB_hab_std"] = float(
            np.sqrt(max(0.0, -2.0 * np.log(max(resultant, 1e-12))))
            * 180.0 / np.pi
        )
        feats["CIELAB_hab_circular_variance"] = 1.0 - resultant

        # 绿度指数: -a* 越大越绿
        feats["CIELAB_greenness"] = float(-a_ch.mean())
        # 黄化指数: b* 越大越黄
        feats["CIELAB_yellowness"] = float(b_ch.mean())

        return feats

    # ----------------------------------------------------------
    # 颜色矩 (在RGB各通道独立计算)
    # ----------------------------------------------------------
    def _extract_color_moments(self, img: np.ndarray,
                               mask: np.ndarray) -> Dict[str, float]:
        """提取RGB三通道颜色矩 (均值/标准差/偏度/峰度)."""
        feats = {}
        ch_names = ["R", "G", "B"]
        for i, name in enumerate(ch_names):
            ch = img[..., i][mask > 0]
            if len(ch) == 0:
                continue

            mean = float(ch.mean())
            std = float(ch.std())

            feats[f"ColorMoment_{name}_mean"] = mean
            feats[f"ColorMoment_{name}_std"] = std

            if std > 1e-10:
                centered = ch - mean
                feats[f"ColorMoment_{name}_skewness"] = float(
                    np.mean(centered**3) / (std**3))
                feats[f"ColorMoment_{name}_kurtosis"] = float(
                    np.mean(centered**4) / (std**4) - 3)
            else:
                feats[f"ColorMoment_{name}_skewness"] = 0.0
                feats[f"ColorMoment_{name}_kurtosis"] = 0.0

        return feats

    # ----------------------------------------------------------
    # 直方图特征
    # ----------------------------------------------------------
    def _extract_histogram_features(self, img_uint8: np.ndarray,
                                    mask: np.ndarray) -> Dict[str, float]:
        """提取RGB直方图特征."""
        feats = {}

        # 灰度直方图
        gray = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2GRAY)
        gray_vals = gray[mask > 0]
        if len(gray_vals) == 0:
            return feats

        hist_feats = histogram_features(gray_vals, bins=self.hist_bins)
        for k, v in hist_feats.items():
            feats[f"Hist_Gray_{k}"] = v

        # RGB各通道直方图
        for i, ch_name in enumerate(["R", "G", "B"]):
            ch_vals = img_uint8[..., i][mask > 0]
            hist_feats = histogram_features(ch_vals, bins=self.hist_bins)
            for k, v in hist_feats.items():
                feats[f"Hist_{ch_name}_{k}"] = v

        return feats

    # ----------------------------------------------------------
    # 色度坐标特征
    # ----------------------------------------------------------
    def _extract_chromaticity_features(self, img: np.ndarray,
                                       mask: np.ndarray) -> Dict[str, float]:
        """提取CIE色度坐标特征."""
        feats = {}

        # Compute XYZ once, then derive both CIE 1931 xyY and CIE 1976 u'v'.
        xyz = rgb_to_xyz(img)
        X_img, Y_img, Z_img = xyz[..., 0], xyz[..., 1], xyz[..., 2]
        total = X_img + Y_img + Z_img + 1e-10
        x_img = X_img / total
        y_img = Y_img / total
        for name, ch in [
            ("x", x_img), ("y", y_img), ("luminance", Y_img)
        ]:
            vals = ch[mask > 0]
            if len(vals) == 0:
                continue
            feats[f"Chromaticity_xyY_{name}_mean"] = float(vals.mean())
            feats[f"Chromaticity_xyY_{name}_std"] = float(vals.std())

        # CIE 1976 u'v'
        uv_denom = X_img + 15 * Y_img + 3 * Z_img + 1e-10
        u_img = 4 * X_img / uv_denom
        v_img = 9 * Y_img / uv_denom
        for name, ch in [("u_prime", u_img), ("v_prime", v_img)]:
            vals = ch[mask > 0]
            if len(vals) == 0:
                continue
            feats[f"Chromaticity_uv_{name}_mean"] = float(vals.mean())
            feats[f"Chromaticity_uv_{name}_std"] = float(vals.std())

        return feats

    # ----------------------------------------------------------
    # 额外工具: 计算两个样本之间的色差
    # ----------------------------------------------------------
    @staticmethod
    def leaf_color_difference(lab1_mean: np.ndarray,
                              lab2_mean: np.ndarray) -> Dict[str, float]:
        """计算两片叶子的CIELAB色差 (用于品种间比较).

        Args:
            lab1_mean: 叶片1的Lab均值 (3,)
            lab2_mean: 叶片2的Lab均值 (3,)

        Returns:
            {"delta_E76": ..., "delta_L": ..., "delta_a": ..., "delta_b": ...}
        """
        dL = lab1_mean[0] - lab2_mean[0]
        da = lab1_mean[1] - lab2_mean[1]
        db = lab1_mean[2] - lab2_mean[2]
        dE = np.sqrt(dL**2 + da**2 + db**2)
        return {
            "delta_E76": float(dE),
            "delta_L": float(dL),
            "delta_a": float(da),
            "delta_b": float(db),
        }
