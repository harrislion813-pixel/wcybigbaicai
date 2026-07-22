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
            fill_value = int(np.median(gray[mask_crop]))
            gray = gray.copy()
            gray[~mask_crop] = fill_value

        # 降级加速 (256 → 64 灰度级)
        if self.levels < 256:
            gray = (gray.astype(np.float64) * (self.levels - 1) / 255).astype(np.uint8)

        # 计算GLCM
        glcm = graycomatrix(
            gray,
            distances=self.distances,
            angles=self.angles_rad,
            levels=self.levels,
            symmetric=True,
            normed=True,
        )  # shape: (levels, levels, n_distances, n_angles)

        # 提取属性
        feats: Dict[str, float] = {}

        for prop in self.properties:
            try:
                prop_array = graycoprops(glcm, prop)
                # shape: (n_distances, n_angles)
                feats[f"GLCM_{prop}_mean"] = float(prop_array.mean())
                feats[f"GLCM_{prop}_std"] = float(prop_array.std())

                # 各距离汇总
                for d_idx, d in enumerate(self.distances):
                    vals = prop_array[d_idx, :]  # 该距离下所有角度
                    feats[f"GLCM_{prop}_d{d}_mean"] = float(vals.mean())
            except Exception:
                feats[f"GLCM_{prop}_mean"] = np.nan

        return feats


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

    def compute(self, mask: np.ndarray,
                pixel_scale: Optional[float] = None) -> Dict[str, float]:
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

        # 找最大连通域 (假设最大的就是目标叶片)
        contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return {f"Shape_{f}": np.nan for f in self.features}

        # 取最大轮廓
        contour = max(contours, key=cv2.contourArea)

        # 基础量
        area_px = cv2.contourArea(contour)
        perimeter_px = cv2.arcLength(contour, True)

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
                continue
            feats[f"Uniformity_{name}_std"] = float(ch.std())
            feats[f"Uniformity_CV_{name}"] = float(ch.std() / (ch.mean() + 1e-10))

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
