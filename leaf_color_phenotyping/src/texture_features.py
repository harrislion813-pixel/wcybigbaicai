"""
纹理与形状特征提取。

纹理特征 (GLCM — 灰度共生矩阵):
    对叶片表面的颜色纹理/斑驳程度进行量化.
    关键指标:
        - contrast: 对比度 (局部变化程度, 斑驳叶片更高)
        - dissimilarity: 不相似性
        - homogeneity: 同质性 (均匀叶片更高)
        - energy / ASM: 能量/角二阶矩 (纹理均匀性)
        - correlation: 相关性 (像素间灰度线性依赖)

形状特征:
    叶片面积、周长、圆度、偏心率等形态参数
"""
from typing import Dict, List, Optional, Tuple
import numpy as np
import cv2
from skimage.feature import graycomatrix, graycoprops


class GLCMTextureExtractor:
    """基于灰度共生矩阵 (GLCM) 的纹理特征提取器.

    计算叶片表面的颜色纹理特征, 用于量化:
        - 叶面颜色均匀性 (homogeneity, energy)
        - 斑点/斑驳程度 (contrast, dissimilarity)
        - 叶脉纹理 (correlation)

    默认在多个距离和角度上计算, 取均值汇总.
    """

    def __init__(self,
                 distances: List[int] = None,
                 angles: List[float] = None,
                 levels: int = 64,
                 properties: List[str] = None):
        """
        Args:
            distances: GLCM像素距离列表
            angles: 角度列表 (度), 0=水平, 90=垂直
            levels: 灰度级数 (通常256, 可降为64加速)
            properties: 要计算的属性列表
        """
        self.distances = [1, 3, 5] if distances is None else list(distances)
        self.angles_deg = [0, 45, 90, 135] if angles is None else list(angles)
        self.angles_rad = np.deg2rad(self.angles_deg)
        self.levels = levels
        default_properties = [
            "contrast", "dissimilarity", "homogeneity",
            "energy", "correlation", "ASM"
        ]
        self.properties = default_properties if properties is None else list(properties)
        if not self.distances or any(
            not isinstance(distance, int) or distance <= 0
            for distance in self.distances
        ):
            raise ValueError("GLCM distances must be positive integers")
        if not isinstance(self.levels, int) or not 2 <= self.levels <= 256:
            raise ValueError("GLCM levels must be an integer in [2, 256]")
        valid_properties = set(default_properties)
        unknown_properties = sorted(set(self.properties) - valid_properties)
        if unknown_properties:
            raise ValueError(f"Unknown GLCM properties: {unknown_properties}")

    def compute(self, img_rgb: np.ndarray,
                mask: Optional[np.ndarray] = None) -> Dict[str, float]:
        """计算GLCM纹理特征.

        Args:
            img_rgb: (H,W,3) uint8 [0,255]
            mask: (H,W) 叶片掩膜

        Returns:
            扁平特征字典
        """
        if not self.properties:
            return {}

        if img_rgb.dtype != np.uint8:
            img_uint8 = (img_rgb * 255).clip(0, 255).astype(np.uint8)
        else:
            img_uint8 = img_rgb

        # 灰度化
        gray = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2GRAY)

        # 掩膜应用
        mask_crop = None
        if mask is not None:
            if mask.max() > 1:
                mask_bin = mask > 127
            else:
                mask_bin = mask > 0.5
            if not np.any(mask_bin):
                return {
                    f"GLCM_{prop}_{suffix}": np.nan
                    for prop in self.properties
                    for suffix in ("mean", "std")
                }

            x, y, w, h = cv2.boundingRect(mask_bin.astype(np.uint8))
            gray = gray[y:y + h, x:x + w].copy()
            mask_crop = mask_bin[y:y + h, x:x + w]

        # 降级加速 (256 → 64 灰度级)
        if self.levels < 256:
            gray = (gray.astype(np.float64) * (self.levels - 1) / 255).astype(np.uint8)

        # 计算GLCM
        if mask_crop is None:
            glcm = graycomatrix(
                gray,
                distances=self.distances,
                angles=self.angles_rad,
                levels=self.levels,
                symmetric=True,
                normed=True,
            )
            valid_pairs = np.ones(
                (len(self.distances), len(self.angles_rad)), dtype=bool
            )
        else:
            glcm, valid_pairs = self._masked_graycomatrix(gray, mask_crop)

        # 提取属性
        feats: Dict[str, float] = {}

        for prop in self.properties:
            try:
                prop_array = graycoprops(glcm, prop)
                # shape: (n_distances, n_angles)
                prop_array = prop_array.astype(np.float64)
                prop_array[~valid_pairs] = np.nan
                finite_values = prop_array[np.isfinite(prop_array)]
                feats[f"GLCM_{prop}_mean"] = (
                    float(finite_values.mean()) if finite_values.size else np.nan
                )
                feats[f"GLCM_{prop}_std"] = (
                    float(finite_values.std()) if finite_values.size else np.nan
                )

                # 各距离汇总
                for d_idx, d in enumerate(self.distances):
                    vals = prop_array[d_idx, :]  # 该距离下所有角度
                    vals = vals[np.isfinite(vals)]
                    feats[f"GLCM_{prop}_d{d}_mean"] = (
                        float(vals.mean()) if vals.size else np.nan
                    )
            except ValueError:
                feats[f"GLCM_{prop}_mean"] = np.nan

        return feats

    def _masked_graycomatrix(
        self, gray: np.ndarray, mask: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Build GLCMs using only pixel pairs whose endpoints are both in-mask."""
        n_distances = len(self.distances)
        n_angles = len(self.angles_rad)
        glcm = np.zeros(
            (self.levels, self.levels, n_distances, n_angles), dtype=np.float64
        )
        valid_pairs = np.zeros((n_distances, n_angles), dtype=bool)
        height, width = gray.shape

        for d_idx, distance in enumerate(self.distances):
            for a_idx, angle in enumerate(self.angles_rad):
                dx = int(round(np.cos(angle) * distance))
                dy = int(round(np.sin(angle) * distance))
                if abs(dx) >= width or abs(dy) >= height:
                    continue

                if dx >= 0:
                    src_x, dst_x = slice(0, width - dx), slice(dx, width)
                else:
                    src_x, dst_x = slice(-dx, width), slice(0, width + dx)
                if dy >= 0:
                    src_y, dst_y = slice(0, height - dy), slice(dy, height)
                else:
                    src_y, dst_y = slice(-dy, height), slice(0, height + dy)

                pair_mask = mask[src_y, src_x] & mask[dst_y, dst_x]
                if not np.any(pair_mask):
                    continue

                source = gray[src_y, src_x][pair_mask].astype(np.int64)
                target = gray[dst_y, dst_x][pair_mask].astype(np.int64)
                counts = np.bincount(
                    source * self.levels + target,
                    minlength=self.levels * self.levels,
                ).reshape(self.levels, self.levels).astype(np.float64)
                counts += counts.T
                total = counts.sum()
                if total <= 0:
                    continue
                glcm[:, :, d_idx, a_idx] = counts / total
                valid_pairs[d_idx, a_idx] = True

        return glcm, valid_pairs


class LeafShapeExtractor:
    """叶片形状特征提取器.

    从分割掩膜中提取叶片形态参数:
        - area: 叶片面积 (像素数)
        - perimeter: 叶片周长
        - circularity: 圆度 = 4πA/P² (1=正圆, <1=不规则)
        - eccentricity: 偏心率 (0=圆, 1=线)
        - solidity: 坚实度 = A/convex_hull_A
        - extent: 范围比 = A/bounding_box_A
        - major_axis_length: 长轴长度
        - minor_axis_length: 短轴长度
        - aspect_ratio: 长宽比
        - roundness: 圆度 (4A)/(π*major²)
    """

    def __init__(self, features: Optional[List[str]] = None):
        default_features = [
            "area", "perimeter", "circularity", "eccentricity",
            "solidity", "extent", "aspect_ratio", "roundness",
            "major_axis_length", "minor_axis_length"
        ]
        self.features = default_features if features is None else list(features)
        unknown_features = sorted(set(self.features) - set(default_features))
        if unknown_features:
            raise ValueError(f"Unknown shape features: {unknown_features}")

    def compute(self, mask: np.ndarray,
                pixel_scale: Optional[float] = None,
                component_policy: str = "largest") -> Dict[str, float]:
        """计算叶片形状特征.

        Args:
            mask: (H,W) uint8 二值掩膜
            pixel_scale: 每个像素对应的实际尺寸 (mm/pixel).
                         若提供, 面积/周长/轴长将转换为实际单位.

        Returns:
            形状特征字典
        """
        if not self.features:
            return {}

        if mask.max() > 1:
            mask_bin = (mask > 127).astype(np.uint8)
        else:
            mask_bin = mask.astype(np.uint8)

        if component_policy not in {"largest", "all"}:
            raise ValueError("component_policy must be 'largest' or 'all'")

        contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return {f"Shape_{f}": np.nan for f in self.features}

        if component_policy == "largest":
            selected_contours = [max(contours, key=cv2.contourArea)]
        else:
            selected_contours = contours
        contour = np.vstack(selected_contours)

        # 基础量
        area_px = sum(cv2.contourArea(item) for item in selected_contours)
        perimeter_px = sum(cv2.arcLength(item, True) for item in selected_contours)

        feats: Dict[str, float] = {}

        # 面积
        scale = pixel_scale or 1.0
        feats["Shape_area"] = area_px * scale ** 2 if pixel_scale else area_px
        feats["Shape_perimeter"] = perimeter_px * scale if pixel_scale else perimeter_px

        # 圆度
        if perimeter_px > 0:
            feats["Shape_circularity"] = 4 * np.pi * area_px / (perimeter_px ** 2)
        else:
            feats["Shape_circularity"] = 0.0

        # 最小外接矩形
        rect = cv2.minAreaRect(contour)
        (cx, cy), (width, height), angle = rect
        major = max(width, height)
        minor = min(width, height)
        feats["Shape_major_axis_length"] = major * scale if pixel_scale else major
        feats["Shape_minor_axis_length"] = minor * scale if pixel_scale else minor
        feats["Shape_aspect_ratio"] = major / (minor + 1e-10)
        feats["Shape_roundness"] = (4 * area_px) / (np.pi * major ** 2 + 1e-10)

        # 偏心率 (通过椭圆拟合)
        if len(contour) >= 5:
            ellipse = cv2.fitEllipse(contour)
            (ex, ey), (ema, emi), eang = ellipse
            major_axis = max(ema, emi)
            minor_axis = min(ema, emi)
            if major_axis > 0:
                ratio = (minor_axis ** 2) / (major_axis ** 2)
                feats["Shape_eccentricity"] = float(np.sqrt(max(0.0, 1 - ratio)))
            else:
                feats["Shape_eccentricity"] = 0.0
        else:
            feats["Shape_eccentricity"] = 0.0

        # 凸包 → 坚实度
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        feats["Shape_solidity"] = area_px / (hull_area + 1e-10)

        # 外接矩形 → extent
        x, y, w, h = cv2.boundingRect(contour)
        feats["Shape_extent"] = area_px / (w * h + 1e-10)

        # 只返回请求的特征
        return {k: feats[k] for k in feats if k in [f"Shape_{f}" for f in self.features]}


class ColorTextureAnalyzer:
    """颜色纹理综合分析器.

    将颜色特征和纹理特征结合, 输出叶片颜色均匀性评估量.
    """

    @staticmethod
    def color_uniformity(lab_img: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
        """计算叶片颜色均匀性指标.

        原理: 在CIELAB空间计算像素间色差(ΔE)的分布,
              色差标准差小 → 颜色均匀 (深绿)
              色差标准差大 → 颜色不均匀 (斑驳/黄化不均)

        Returns:
            {
                "Uniformity_L_std": L通道标准差 (明度不均匀性),
                "Uniformity_a_std": a通道标准差 (红绿不均匀性),
                "Uniformity_b_std": b通道标准差 (黄蓝不均匀性),
                "Uniformity_dE_mean": 每像素与均值的平均色差,
                "Uniformity_dE_std": 色差标准差 (总不均匀性指标),
                "Uniformity_CV_L": L通道变异系数,
            }
        """
        if mask.max() > 1:
            mask_bin = mask > 127
        else:
            mask_bin = mask > 0.5

        feats = {}
        for i, name in enumerate(["L", "a", "b"]):
            ch = lab_img[..., i][mask_bin]
            if len(ch) == 0:
                feats[f"Uniformity_{name}_std"] = np.nan
                feats[f"Uniformity_CV_{name}"] = np.nan
                feats[f"Uniformity_{name}_MAD"] = np.nan
                continue
            feats[f"Uniformity_{name}_std"] = float(ch.std())
            mean_abs = abs(float(ch.mean()))
            feats[f"Uniformity_CV_{name}"] = (
                float(ch.std() / mean_abs) if mean_abs > 1.0 else np.nan
            )
            median = float(np.median(ch))
            feats[f"Uniformity_{name}_MAD"] = float(np.median(np.abs(ch - median)))

        # 像素级色差 (逐像素与均值Lab的ΔE76)
        if lab_img[mask_bin].size > 0:
            mean_lab = lab_img[mask_bin].mean(axis=0)
            diffs = lab_img[mask_bin].astype(np.float64) - mean_lab
            dE_pixels = np.sqrt((diffs ** 2).sum(axis=1))
            feats["Uniformity_dE_mean"] = float(dE_pixels.mean())
            feats["Uniformity_dE_std"] = float(dE_pixels.std())
        else:
            feats["Uniformity_dE_mean"] = np.nan
            feats["Uniformity_dE_std"] = np.nan

        return feats
