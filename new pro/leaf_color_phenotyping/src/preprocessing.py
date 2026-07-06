"""
图像预处理：RAW转换、白平衡、颜色校准（基于ColorChecker色卡）。

核心流程:
    RAW → 线性RGB → 白平衡 → 颜色校正矩阵(CCM) → 目标色彩空间
"""
from pathlib import Path
from typing import Optional, Tuple, Dict, List

import numpy as np
import cv2

from .utils import (
    COLORCHECKER_24_LAB_D50, rgb_to_lab, delta_e_76, safe_mkdir
)


class ImagePreprocessor:
    """图像预处理器 — 处理RAW转RGB、颜色校准等."""

    def __init__(self,
                 target_illuminant: str = "D65",
                 calibration_method: str = "polynomial",
                 polynomial_degree: int = 2):
        """
        Args:
            target_illuminant: 目标光源 ("D50", "D65", "A")
            calibration_method: 颜色校正方法
                - "linear": 3×3 线性回归
                - "polynomial": 多项式回归 (推荐)
                - "root_polynomial": 根多项式回归
            polynomial_degree: 多项式阶数 (method="polynomial" 时有效)
        """
        self.target_illuminant = target_illuminant
        self.calibration_method = calibration_method
        self.polynomial_degree = polynomial_degree
        self._ccm: Optional[np.ndarray] = None  # 颜色校正矩阵

    # ----------------------------------------------------------
    # RAW 文件读取 (依赖 rawpy)
    # ----------------------------------------------------------
    @staticmethod
    def read_raw(raw_path: str,
                 use_camera_wb: bool = False,
                 output_bps: int = 16) -> np.ndarray:
        """读取RAW文件并转换为线性RGB numpy数组.

        Args:
            raw_path: RAW文件路径 (.cr2, .nef, .arw, .dng, .raw)
            use_camera_wb: 是否使用相机白平衡系数
            output_bps: 输出位深度 (8 / 16)

        Returns:
            float32 RGB图像, shape (H, W, 3), range [0, 1]
        """
        try:
            import rawpy
        except ImportError:
            raise ImportError(
                "读取RAW文件需要安装 rawpy: pip install rawpy"
            )

        with rawpy.imread(str(raw_path)) as raw:
            # 后处理参数
            # output_color=rawpy.ColorSpace.sRGB → 使用相机内建色彩矩阵
            # 若需纯线性数据, 用 output_color=rawpy.ColorSpace.raw
            rgb = raw.postprocess(
                use_camera_wb=use_camera_wb,
                output_color=rawpy.ColorSpace.sRGB,
                gamma=(1, 1),           # 线性输出 (不做gamma)
                no_auto_bright=True,
                output_bps=16 if output_bps == 16 else 8,
            )

        rgb_float = rgb.astype(np.float32) / (2**output_bps - 1)
        return np.clip(rgb_float, 0, 1)

    @staticmethod
    def read_image(image_path: str) -> np.ndarray:
        """读取普通图像文件 (PNG/JPG/TIFF).

        Returns:
            float32 RGB, shape (H, W, 3), range [0, 1]
        """
        bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(f"无法读取图像: {image_path}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return rgb.astype(np.float32) / 255.0

    # ----------------------------------------------------------
    # 白平衡 (灰度世界 / 完美反射 / 灰卡)
    # ----------------------------------------------------------
    @staticmethod
    def white_balance_gray_world(img_rgb: np.ndarray) -> np.ndarray:
        """灰度世界白平衡.

        假设场景平均反射率为灰色 → R、G、B 均值应相等.
        """
        r_mean = img_rgb[..., 0].mean()
        g_mean = img_rgb[..., 1].mean()
        b_mean = img_rgb[..., 2].mean()
        gray = (r_mean + g_mean + b_mean) / 3

        result = img_rgb.copy()
        result[..., 0] *= gray / (r_mean + 1e-10)
        result[..., 1] *= gray / (g_mean + 1e-10)
        result[..., 2] *= gray / (b_mean + 1e-10)
        return np.clip(result, 0, 1)

    @staticmethod
    def white_balance_perfect_reflector(img_rgb: np.ndarray,
                                        percentile: float = 99.9) -> np.ndarray:
        """完美反射白平衡.

        假设图像中最亮的像素应为白色.
        """
        result = img_rgb.copy()
        for c in range(3):
            thresh = np.percentile(img_rgb[..., c], percentile)
            result[..., c] /= (thresh + 1e-10)
        return np.clip(result, 0, 1)

    @staticmethod
    def white_balance_gray_card(img_rgb: np.ndarray,
                                gray_roi: np.ndarray) -> np.ndarray:
        """基于灰卡的白平衡.

        Args:
            img_rgb: 输入RGB
            gray_roi: 灰卡区域的平均RGB (3,) 数组

        Returns:
            白平衡校正后的图像
        """
        # 以绿色通道为基准归一化
        g_val = gray_roi[1]
        if g_val < 1e-6:
            return img_rgb
        gains = np.array([
            g_val / (gray_roi[0] + 1e-10),
            1.0,
            g_val / (gray_roi[2] + 1e-10),
        ], dtype=np.float32)
        result = img_rgb * gains.reshape(1, 1, 3)
        return np.clip(result, 0, 1)

    # ----------------------------------------------------------
    # 颜色校正矩阵 (CCM) 计算
    # ----------------------------------------------------------
    def compute_color_correction_matrix(
        self,
        measured_rgb: np.ndarray,      # (N, 3) — 图像中检测到的色块RGB
        reference_lab: np.ndarray = None,  # (N, 3) — 色卡参考Lab值
    ) -> np.ndarray:
        """由 ColorChecker 色块计算颜色校正矩阵.

        计算流程:
            1. 将 RGB 值展开为多项式特征 (若 polynomial)
            2. 将 reference Lab 转为 RGB (通过标准转换)
            3. 最小二乘求解: M = (X^T X)^-1 X^T Y
            4. 应用 CCM: corrected = expand(rgb) @ M

        Args:
            measured_rgb: 图像中提取的24个色块的 RGB 均值, shape (24, 3)
            reference_lab: 色卡标准 Lab 值, shape (24, 3).
                           默认使用内置 ColorChecker 24 D50 Lab 值.

        Returns:
            CCM: 若 method="linear", shape (3, 3);
                 若 method="polynomial" degree=2, shape (9, 3)
        """
        if reference_lab is None:
            reference_lab = COLORCHECKER_24_LAB_D50

        # 参考Lab → 目标RGB (简化: 使用标准sRGB的前向模型)
        # 实际使用中建议用 colour-science 库做 Lab→XYZ→RGB 的精确转换
        ref_rgb = self._lab_to_srgb_approx(reference_lab)

        # ---- 确保色块数量一致 ----
        n_patches = min(len(measured_rgb), len(ref_rgb))
        src = measured_rgb[:n_patches]
        dst = ref_rgb[:n_patches]

        if self.calibration_method == "linear":
            A = src  # (N, 3)
            # 最小二乘: M = A^-1 @ dst
            M, _, _, _ = np.linalg.lstsq(A, dst, rcond=None)
            self._ccm = M  # (3, 3)
        elif self.calibration_method == "polynomial":
            A = self._expand_polynomial(src, self.polynomial_degree)
            M, _, _, _ = np.linalg.lstsq(A, dst, rcond=None)
            self._ccm = M  # (K, 3)
        else:
            raise ValueError(f"Unknown calibration method: {self.calibration_method}")

        print(f"[ColorCalibration] CCM computed, shape={self._ccm.shape}")
        return self._ccm

    def apply_color_correction(self, img_rgb: np.ndarray) -> np.ndarray:
        """对图像应用已计算的CCM.

        Args:
            img_rgb: shape (H, W, 3), float32, range [0, 1]

        Returns:
            校正后的 RGB 图像
        """
        if self._ccm is None:
            print("WARNING: CCM not computed yet, returning original image.")
            return img_rgb

        h, w = img_rgb.shape[:2]
        pixels = img_rgb.reshape(-1, 3)

        if self.calibration_method == "linear":
            corrected = pixels @ self._ccm
        elif self.calibration_method == "polynomial":
            expanded = self._expand_polynomial(pixels, self.polynomial_degree)
            corrected = expanded @ self._ccm
        else:
            corrected = pixels

        corrected = np.clip(corrected, 0, 1)
        return corrected.reshape(h, w, 3).astype(np.float32)

    # ----------------------------------------------------------
    # 完整预处理流水线
    # ----------------------------------------------------------
    def process(self,
                image_path: str,
                white_balance_method: str = "gray_world",
                gray_roi: Optional[np.ndarray] = None,
                apply_ccm: bool = True) -> Dict[str, np.ndarray]:
        """完整的图像预处理流水线.

        Args:
            image_path: 图像路径
            white_balance_method: "gray_world" | "perfect_reflector" | "gray_card" | "none"
            gray_roi: white_balance_method="gray_card" 时需提供灰卡ROI的RGB均值
            apply_ccm: 是否应用已计算的CCM

        Returns:
            {"rgb": ..., "lab": ..., "hsv": ...} 预处理后的多色彩空间图像
        """
        # Step 1: 读取
        ext = Path(image_path).suffix.lower()
        if ext in (".raw", ".dng", ".cr2", ".nef", ".arw"):
            img = self.read_raw(image_path)
        else:
            img = self.read_image(image_path)

        # Step 2: 白平衡
        if white_balance_method == "gray_world":
            img = self.white_balance_gray_world(img)
        elif white_balance_method == "perfect_reflector":
            img = self.white_balance_perfect_reflector(img)
        elif white_balance_method == "gray_card" and gray_roi is not None:
            img = self.white_balance_gray_card(img, gray_roi)

        # Step 3: 颜色校正
        if apply_ccm and self._ccm is not None:
            img = self.apply_color_correction(img)

        # Step 4: 多色彩空间输出
        img_uint8 = (img * 255).clip(0, 255).astype(np.uint8)

        return {
            "rgb": img,
            "rgb_uint8": img_uint8,
            "hsv": cv2.cvtColor(cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR),
                                cv2.COLOR_BGR2HSV),
            "lab": rgb_to_lab(img),
        }

    # ----------------------------------------------------------
    # 辅助方法
    # ----------------------------------------------------------
    @staticmethod
    def _expand_polynomial(rgb: np.ndarray, degree: int = 2) -> np.ndarray:
        """将 RGB 向量展开为多项式特征.

        degree=2: [R, G, B, R², G², B², RG, RB, GB], shape (N, 9)
        degree=3: 再加三次项 (共 19 项)
        """
        R, G, B = rgb[..., 0], rgb[..., 1], rgb[..., 2]
        terms = [R, G, B]

        if degree >= 2:
            terms.extend([R**2, G**2, B**2, R*G, R*B, G*B])
        if degree >= 3:
            terms.extend([R**3, G**3, B**3,
                          R**2*G, R**2*B, G**2*R, G**2*B, B**2*R, B**2*G,
                          R*G*B])

        return np.column_stack(terms)

    @staticmethod
    def _lab_to_srgb_approx(lab: np.ndarray) -> np.ndarray:
        """Lab → sRGB 近似转换 (用于CCM计算).

        精确转换建议使用 colour-science 库.
        """
        # Lab → XYZ (D50 白点)
        L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]

        fy = (L + 16) / 116
        fx = a / 500 + fy
        fz = fy - b / 200

        delta = 6 / 29
        xyz = np.zeros_like(lab)
        for i, f in enumerate([fx, fy, fz]):
            mask = f > delta
            arr = np.where(mask, f ** 3, (f - 4/29) * 3 * delta**2)
            xyz[..., i] = arr

        # D50 白点归一化
        xyz[..., 0] *= 0.96422
        xyz[..., 1] *= 1.00000
        xyz[..., 2] *= 0.82521

        # XYZ → linear sRGB
        M = np.array([
            [3.2404542, -1.5371385, -0.4985314],
            [-0.9692660, 1.8760108, 0.0415560],
            [0.0556434, -0.2040259, 1.0572252],
        ])
        flat_xyz = xyz.reshape(-1, 3).T
        linear_rgb = (M @ flat_xyz).T.reshape(xyz.shape)

        # Gamma 校正
        mask = linear_rgb <= 0.0031308
        srgb = np.where(mask, 12.92 * linear_rgb,
                        1.055 * linear_rgb ** (1/2.4) - 0.055)
        return np.clip(srgb, 0, 1)
