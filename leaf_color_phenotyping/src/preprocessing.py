"""
图像预处理：RAW转换、白平衡、颜色校准（基于ColorChecker色卡）。

核心流程:
    RAW/JPEG → 显式 RGB 工作域 → 白平衡 → 颜色校正 → 目标色彩空间
"""
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
import json
import warnings

import numpy as np
import cv2

from .utils import (
    COLORCHECKER_24_LAB_D50, delta_e_2000, rgb_to_lab,
    RAW_IMAGE_EXTENSIONS, read_image_rgb, BRADFORD_D50_TO_D65,
    get_colorchecker_reference_lab,
)


@dataclass(frozen=True)
class CalibrationFitResult:
    """Diagnostics produced while fitting a legacy RGB-space CCM."""

    matrix: np.ndarray
    rank: int
    condition_number: float
    residuals: np.ndarray
    training_delta_e00: Dict[str, float]
    excluded_patches: Tuple[str, ...] = ()


class ImagePreprocessor:
    """图像预处理器 — 处理RAW转RGB、颜色校准等."""

    _SUPPORTED_POLYNOMIAL_DEGREES = {1, 2, 3}
    _MAX_CCM_CONDITION_NUMBER = 1e8

    def __init__(self,
                 target_illuminant: str = "D65",
                 calibration_method: str = "polynomial",
                 polynomial_degree: int = 2,
                 raw_use_camera_wb: bool = True,
                 raw_output_bps: int = 16,
                 working_domain: str = "encoded_srgb"):
        """
        Args:
            target_illuminant: 目标光源；当前 sRGB/CIELAB 管线仅支持 "D65"
            calibration_method: 颜色校正方法
                - "linear": 3×3 线性回归
                - "polynomial": 多项式回归 (推荐)
            polynomial_degree: 多项式阶数 (method="polynomial" 时有效)
            raw_use_camera_wb: RAW 解码时是否使用相机白平衡
            raw_output_bps: RAW 后处理位深，8 或 16
            working_domain: encoded_srgb、linear_srgb 或 camera_linear_rgb
        """
        if str(target_illuminant).upper() != "D65":
            raise ValueError("target_illuminant currently supports only D65")
        if calibration_method not in {"linear", "polynomial"}:
            raise ValueError(f"Unknown calibration method: {calibration_method}")
        if calibration_method == "polynomial":
            self._validate_polynomial_degree(polynomial_degree)
        self.target_illuminant = "D65"
        self.calibration_method = calibration_method
        self.polynomial_degree = polynomial_degree
        self.raw_use_camera_wb = bool(raw_use_camera_wb)
        if raw_output_bps not in (8, 16):
            raise ValueError("raw_output_bps must be 8 or 16")
        self.raw_output_bps = raw_output_bps
        if working_domain not in {
            "encoded_srgb", "linear_srgb", "camera_linear_rgb",
        }:
            raise ValueError(f"Unknown working_domain: {working_domain}")
        self.working_domain = working_domain
        self._ccm: Optional[np.ndarray] = None  # 颜色校正矩阵
        self.last_calibration_report: Optional[CalibrationFitResult] = None

    @property
    def has_color_correction_matrix(self) -> bool:
        """Whether a structurally valid color-correction matrix is available."""
        return self._ccm is not None

    @classmethod
    def _validate_polynomial_degree(cls, degree: int) -> int:
        if isinstance(degree, bool) or not isinstance(degree, (int, np.integer)):
            raise TypeError("polynomial_degree must be an integer")
        degree = int(degree)
        if degree not in cls._SUPPORTED_POLYNOMIAL_DEGREES:
            supported = sorted(cls._SUPPORTED_POLYNOMIAL_DEGREES)
            raise ValueError(f"polynomial_degree must be one of {supported}")
        return degree

    def set_color_correction_matrix(self, matrix: np.ndarray) -> np.ndarray:
        """Validate and install a precomputed color-correction matrix."""
        matrix = np.asarray(matrix, dtype=np.float64)
        if self.calibration_method == "linear":
            expected_shape = (3, 3)
        elif self.calibration_method == "polynomial":
            n_terms = self._expand_polynomial(
                np.zeros((1, 3), dtype=np.float64), self.polynomial_degree
            ).shape[1]
            expected_shape = (n_terms, 3)
        else:
            raise ValueError(f"Unknown calibration method: {self.calibration_method}")

        if matrix.shape != expected_shape:
            raise ValueError(
                f"CCM shape {matrix.shape} does not match {self.calibration_method} "
                f"calibration; expected {expected_shape}"
            )
        if not np.isfinite(matrix).all():
            raise ValueError("CCM contains NaN or infinite values")
        self._ccm = matrix
        return self._ccm

    def load_color_correction_matrix(self, path: str) -> np.ndarray:
        """Load a CCM from .npy, JSON, YAML, or delimited text."""
        ccm_path = Path(path)
        if not ccm_path.is_file():
            raise FileNotFoundError(f"颜色校正矩阵文件不存在: {ccm_path}")

        suffix = ccm_path.suffix.lower()
        if suffix == ".npy":
            matrix = np.load(ccm_path, allow_pickle=False)
        elif suffix == ".json":
            with ccm_path.open("r", encoding="utf-8") as handle:
                matrix = json.load(handle)
        elif suffix in (".yaml", ".yml"):
            try:
                import yaml
            except ImportError as exc:
                raise ImportError("读取 YAML CCM 需要安装 PyYAML") from exc
            with ccm_path.open("r", encoding="utf-8") as handle:
                matrix = yaml.safe_load(handle)
        else:
            matrix = np.loadtxt(ccm_path, delimiter="," if suffix == ".csv" else None)

        return self.set_color_correction_matrix(matrix)

    # ----------------------------------------------------------
    # RAW 文件读取 (依赖 rawpy)
    # ----------------------------------------------------------
    @staticmethod
    def read_raw(raw_path: str,
                 use_camera_wb: bool = True,
                 output_bps: int = 16,
                 linear_output: bool = False,
                 output_color: str = "srgb") -> np.ndarray:
        """读取RAW文件并转换为标准sRGB numpy数组.

        Args:
            raw_path: RAW文件路径 (.cr2, .nef, .arw, .raf, .dng, .raw)
            use_camera_wb: 是否使用相机白平衡系数
            output_bps: 输出位深度 (8 / 16)
            linear_output: 是否返回线性sRGB；默认False以匹配JPEG和下游颜色转换
            output_color: rawpy 输出空间，"srgb" 或 "raw"

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
            gamma = (1, 1) if linear_output else (2.222, 4.5)
            if output_color == "srgb":
                rawpy_color_space = rawpy.ColorSpace.sRGB
            elif output_color == "raw":
                rawpy_color_space = rawpy.ColorSpace.raw
            else:
                raise ValueError("output_color must be 'srgb' or 'raw'")
            rgb = raw.postprocess(
                use_camera_wb=use_camera_wb,
                output_color=rawpy_color_space,
                gamma=gamma,
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
        return read_image_rgb(image_path, as_float=True)

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
        reference_id: Optional[str] = None,
        return_report: bool = False,
    ) -> Union[np.ndarray, CalibrationFitResult]:
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
        if reference_lab is not None and reference_id is not None:
            raise ValueError("Provide reference_lab or reference_id, not both")
        if reference_id is not None:
            reference_lab = get_colorchecker_reference_lab(reference_id)
        elif reference_lab is None:
            warnings.warn(
                "Implicit ColorChecker reference uses the pre-November-2014 "
                "chart. Pass reference_id explicitly for reproducible calibration.",
                FutureWarning,
                stacklevel=2,
            )
            reference_lab = COLORCHECKER_24_LAB_D50

        src, reference_lab = self._validate_calibration_samples(
            measured_rgb, reference_lab
        )

        # 参考Lab → 目标RGB (简化: 使用标准sRGB的前向模型)
        # 实际使用中建议用 colour-science 库做 Lab→XYZ→RGB 的精确转换
        ref_rgb = self._lab_to_srgb_approx(reference_lab)

        if self.calibration_method == "linear":
            A = src  # (N, 3)
        elif self.calibration_method == "polynomial":
            A = self._expand_polynomial(src, self.polynomial_degree)
        else:
            raise ValueError(f"Unknown calibration method: {self.calibration_method}")

        n_terms = A.shape[1]
        minimum_patches = 2 * n_terms
        if len(src) < minimum_patches:
            raise ValueError(
                f"CCM fitting requires at least {minimum_patches} patches for "
                f"{n_terms} model terms; received {len(src)}"
            )

        rank = int(np.linalg.matrix_rank(A))
        if rank != n_terms:
            raise ValueError(
                f"CCM design matrix is rank deficient: rank={rank}, terms={n_terms}"
            )
        condition_number = float(np.linalg.cond(A))
        if not np.isfinite(condition_number):
            raise ValueError("CCM design matrix condition number is not finite")
        if condition_number > self._MAX_CCM_CONDITION_NUMBER:
            raise ValueError(
                "CCM design matrix is ill-conditioned: "
                f"condition_number={condition_number:.6g} exceeds "
                f"{self._MAX_CCM_CONDITION_NUMBER:.6g}"
            )

        M, residuals, fitted_rank, _ = np.linalg.lstsq(A, ref_rgb, rcond=None)
        if int(fitted_rank) != n_terms:
            raise ValueError(
                f"CCM least-squares fit lost rank: rank={fitted_rank}, terms={n_terms}"
            )
        self.set_color_correction_matrix(M)

        fitted_rgb = np.clip(A @ self._ccm, 0, 1)
        fitted_lab = rgb_to_lab(fitted_rgb.reshape(-1, 1, 3))[:, 0, :]
        reference_lab_d65 = rgb_to_lab(ref_rgb.reshape(-1, 1, 3))[:, 0, :]
        delta_e = np.asarray(
            delta_e_2000(fitted_lab, reference_lab_d65), dtype=np.float64
        )
        delta_metrics = {
            "mean": float(np.mean(delta_e)),
            "median": float(np.median(delta_e)),
            "p95": float(np.percentile(delta_e, 95)),
            "max": float(np.max(delta_e)),
        }
        self.last_calibration_report = CalibrationFitResult(
            matrix=self._ccm.copy(),
            rank=rank,
            condition_number=condition_number,
            residuals=np.asarray(residuals, dtype=np.float64),
            training_delta_e00=delta_metrics,
        )

        print(
            f"[ColorCalibration] CCM computed, shape={self._ccm.shape}, "
            f"rank={rank}, condition={condition_number:.3g}, "
            f"median_dE00={delta_metrics['median']:.3f}"
        )
        if return_report:
            return self.last_calibration_report
        return self._ccm

    @staticmethod
    def _validate_calibration_samples(
        measured_rgb: np.ndarray,
        reference_lab: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Validate paired patch measurements before any regression is attempted."""
        measured = np.asarray(measured_rgb, dtype=np.float64)
        reference = np.asarray(reference_lab, dtype=np.float64)

        if measured.ndim != 2 or measured.shape[1:] != (3,):
            raise ValueError("measured_rgb must have shape (N, 3)")
        if reference.ndim != 2 or reference.shape[1:] != (3,):
            raise ValueError("reference_lab must have shape (N, 3)")
        if len(measured) == 0:
            raise ValueError("calibration patch arrays must not be empty")
        if len(measured) != len(reference):
            raise ValueError(
                "measured_rgb and reference_lab must contain the same number "
                f"of patches; received {len(measured)} and {len(reference)}"
            )
        if not np.isfinite(measured).all():
            raise ValueError("measured_rgb contains NaN or infinite values")
        if not np.isfinite(reference).all():
            raise ValueError("reference_lab contains NaN or infinite values")
        if np.any(measured < 0) or np.any(measured > 1):
            raise ValueError(
                "measured_rgb must be normalized to [0, 1]; declare and "
                "normalize 8/16-bit input before fitting"
            )
        if np.any(reference[:, 0] < 0) or np.any(reference[:, 0] > 100):
            raise ValueError("reference_lab L* values must be in [0, 100]")
        return measured, reference

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

        img_rgb = np.asarray(img_rgb)
        if img_rgb.ndim != 3 or img_rgb.shape[-1] != 3:
            raise ValueError("img_rgb must have shape (H, W, 3)")
        if not np.isfinite(img_rgb).all():
            raise ValueError("img_rgb contains NaN or infinite values")
        if np.any(img_rgb < 0) or np.any(img_rgb > 1):
            raise ValueError("img_rgb must be normalized to [0, 1]")

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
                white_balance_method: str = "none",
                gray_roi: Optional[np.ndarray] = None,
                apply_ccm: bool = True,
                compute_derived: bool = True) -> Dict[str, np.ndarray]:
        """完整的图像预处理流水线.

        Args:
            image_path: 图像路径
            white_balance_method: "gray_world" | "perfect_reflector" | "gray_card" | "none"
            gray_roi: white_balance_method="gray_card" 时需提供灰卡ROI的RGB均值
            apply_ccm: 是否应用已计算的CCM
            compute_derived: 是否同时计算 HSV 和 CIELAB；批处理可在裁剪后再计算

        Returns:
            预处理后的多色彩空间图像；`segmentation_rgb`（可用时）始终是
            CCM 与表型白平衡之前的 encoded-sRGB 分割代理图。
        """
        # Step 1: 读取
        ext = Path(image_path).suffix.lower()
        segmentation_img = None
        if ext in RAW_IMAGE_EXTENSIONS:
            if (
                self.working_domain == "encoded_srgb"
                and self.raw_use_camera_wb
                and self.raw_output_bps == 16
            ):
                # Preserve the historical one-argument call for the default path.
                img = self.read_raw(image_path)
            else:
                img = self.read_raw(
                    image_path,
                    use_camera_wb=self.raw_use_camera_wb,
                    output_bps=self.raw_output_bps,
                    linear_output=self.working_domain != "encoded_srgb",
                    output_color=(
                        "raw" if self.working_domain == "camera_linear_rgb" else "srgb"
                    ),
                )
            if self.working_domain == "encoded_srgb":
                segmentation_img = img.copy()
        else:
            if self.working_domain == "camera_linear_rgb":
                raise ValueError(
                    "camera_linear_rgb calibration profiles require RAW input images"
                )
            img = self.read_image(image_path)
            segmentation_img = img.copy()
            if self.working_domain == "linear_srgb":
                img = self._decode_srgb(img)

        # Step 2: 白平衡
        if white_balance_method == "gray_world":
            img = self.white_balance_gray_world(img)
        elif white_balance_method == "perfect_reflector":
            img = self.white_balance_perfect_reflector(img)
        elif white_balance_method == "gray_card":
            if gray_roi is None:
                raise ValueError("white_balance_method='gray_card' requires gray_roi RGB values")
            gray_roi = np.asarray(gray_roi, dtype=np.float32)
            if gray_roi.shape != (3,) or not np.isfinite(gray_roi).all():
                raise ValueError("gray_roi must contain exactly three finite RGB values")
            img = self.white_balance_gray_card(img, gray_roi)
        elif white_balance_method != "none":
            raise ValueError(f"Unknown white balance method: {white_balance_method}")

        # Step 3: 颜色校正
        if apply_ccm and self._ccm is not None:
            img = self.apply_color_correction(img)

        # Step 4: 多色彩空间输出
        img_uint8 = (img * 255).clip(0, 255).astype(np.uint8)

        result = {
            "rgb": img,
            "rgb_uint8": img_uint8,
        }
        if segmentation_img is not None:
            result["segmentation_rgb"] = segmentation_img
        if compute_derived:
            result["hsv"] = cv2.cvtColor(
                cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR), cv2.COLOR_BGR2HSV
            )
            result["lab"] = rgb_to_lab(img)
        return result

    def process_segmentation_image(self, image_path: str) -> np.ndarray:
        """Decode an encoded-sRGB proxy that is independent of phenotype CCM."""
        ext = Path(image_path).suffix.lower()
        if ext in RAW_IMAGE_EXTENSIONS:
            return self.read_raw(
                image_path,
                use_camera_wb=self.raw_use_camera_wb,
                output_bps=self.raw_output_bps,
                linear_output=False,
                output_color="srgb",
            )
        return self.read_image(image_path)

    # ----------------------------------------------------------
    # 辅助方法
    # ----------------------------------------------------------
    @staticmethod
    def _decode_srgb(img_rgb: np.ndarray) -> np.ndarray:
        values = np.asarray(img_rgb, dtype=np.float64)
        linear = np.where(
            values <= 0.04045,
            values / 12.92,
            ((values + 0.055) / 1.055) ** 2.4,
        )
        return linear.astype(np.float32)

    @staticmethod
    def _expand_polynomial(rgb: np.ndarray, degree: int = 2) -> np.ndarray:
        """将 RGB 向量展开为多项式特征.

        degree=2: [R, G, B, R², G², B², RG, RB, GB], shape (N, 9)
        degree=3: 再加三次项 (共 19 项)
        """
        degree = ImagePreprocessor._validate_polynomial_degree(degree)
        rgb = np.asarray(rgb, dtype=np.float64)
        if rgb.ndim != 2 or rgb.shape[1:] != (3,):
            raise ValueError("rgb polynomial input must have shape (N, 3)")
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
        """Convert ColorChecker Lab (D50) values to display-referred sRGB (D65).

        精确转换建议使用 colour-science 库.
        """
        lab = np.asarray(lab, dtype=np.float64)
        if lab.ndim != 2 or lab.shape[1:] != (3,):
            raise ValueError("lab must have shape (N, 3)")

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

        # ColorChecker reference values are D50, while sRGB uses a D65 white.
        # Apply Bradford chromatic adaptation before the D65 XYZ→sRGB matrix.
        flat_xyz = xyz.reshape(-1, 3)
        xyz = (BRADFORD_D50_TO_D65 @ flat_xyz.T).T.reshape(xyz.shape)

        # XYZ (D65) → linear sRGB
        M = np.array([
            [3.2404542, -1.5371385, -0.4985314],
            [-0.9692660, 1.8760108, 0.0415560],
            [0.0556434, -0.2040259, 1.0572252],
        ])
        flat_xyz = xyz.reshape(-1, 3).T
        linear_rgb = (M @ flat_xyz).T.reshape(xyz.shape)

        # Gamma 校正
        mask = linear_rgb <= 0.0031308
        srgb = np.empty_like(linear_rgb)
        srgb[mask] = 12.92 * linear_rgb[mask]
        positive = ~mask
        srgb[positive] = 1.055 * np.power(linear_rgb[positive], 1 / 2.4) - 0.055
        return np.clip(srgb, 0, 1)
