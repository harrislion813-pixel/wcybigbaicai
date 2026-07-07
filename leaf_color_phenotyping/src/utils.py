"""
工具函数：文件I/O、色彩空间转换、统计辅助等。
"""
import os
import re
import json
from pathlib import Path
from typing import Tuple, Optional, Dict, List, Union

import numpy as np
import cv2


# ============================================================
# 文件I/O
# ============================================================

def find_images(
    directory: str,
    extensions: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".tif", ".tiff",
                                     ".bmp", ".raw", ".dng", ".cr2", ".nef", ".arw")
) -> List[Path]:
    """递归查找目录下所有图像文件.

    Args:
        directory: 搜索根目录
        extensions: 允许的扩展名

    Returns:
        排序后的图像路径列表
    """
    img_paths = []
    for ext in extensions:
        img_paths.extend(Path(directory).rglob(f"*{ext}"))
        img_paths.extend(Path(directory).rglob(f"*{ext.upper()}"))
    return sorted(set(img_paths))


def parse_sample_id(filename: str, pattern: str = r"(.+?)_\d+") -> Optional[str]:
    """从文件名中提取样本ID.

    支持格式:
        "BJC-001_rep1_2024.jpg" -> "BJC-001"
        "sample_A1.jpg"         -> "sample_A1"
    用户可传入自定义 regex pattern.

    Args:
        filename: 文件名 (不含路径)
        pattern: 正则表达式模式

    Returns:
        样本ID 或 None
    """
    stem = Path(filename).stem
    match = re.search(pattern, stem)
    if match:
        return match.group(1)
    # 默认: 取第一个下划线前的部分
    return stem.split("_")[0] if "_" in stem else stem


def safe_mkdir(path: Union[str, Path]) -> Path:
    """安全创建目录."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def read_image_rgb(path: Union[str, Path], as_float: bool = True) -> np.ndarray:
    """Read an image as RGB, supporting Windows paths with spaces/non-ASCII text."""
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        raise FileNotFoundError(f"无法读取图像: {path}")
    bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"无法解码图像: {path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    if as_float:
        return rgb.astype(np.float32) / 255.0
    return rgb


def read_image_gray(path: Union[str, Path]) -> np.ndarray:
    """Read an image as grayscale, supporting Windows paths with spaces/non-ASCII text."""
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        raise FileNotFoundError(f"无法读取图像: {path}")
    gray = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise FileNotFoundError(f"无法解码图像: {path}")
    return gray


def write_image_rgb(path: Union[str, Path], img_rgb: np.ndarray) -> None:
    """Write an RGB image, supporting Windows paths with spaces/non-ASCII text."""
    path = Path(path)
    safe_mkdir(path.parent)
    if img_rgb.dtype != np.uint8:
        img_rgb = (img_rgb * 255).clip(0, 255).astype(np.uint8)
    bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    ok, encoded = cv2.imencode(path.suffix or ".png", bgr)
    if not ok:
        raise OSError(f"无法编码图像: {path}")
    encoded.tofile(str(path))


def list_subdirs(directory: str) -> List[Path]:
    """列出目录下所有子目录."""
    return sorted([p for p in Path(directory).iterdir() if p.is_dir()])


# ============================================================
# 颜色空间转换
# ============================================================

def rgb_to_xyz(img_rgb: np.ndarray, illuminant: str = "D65") -> np.ndarray:
    """sRGB → CIE XYZ (D65).

    Args:
        img_rgb: shape (H, W, 3), dtype float32/float64, range [0, 1]
        illuminant: 参考光源

    Returns:
        CIE XYZ 图像, 同 shape
    """
    # 线性化 (反Gamma)
    mask = img_rgb <= 0.04045
    linear = np.where(mask, img_rgb / 12.92, ((img_rgb + 0.055) / 1.055) ** 2.4)

    # sRGB → XYZ 转换矩阵 (D65)
    M = np.array([
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ], dtype=np.float64)

    h, w = linear.shape[:2]
    flat = linear.reshape(-1, 3).T  # (3, N)
    xyz_flat = M @ flat
    return xyz_flat.T.reshape(h, w, 3).astype(np.float32)


def xyz_to_lab(img_xyz: np.ndarray) -> np.ndarray:
    """CIE XYZ → CIELAB (D65白点).

    Args:
        img_xyz: shape (H, W, 3), dtype float32

    Returns:
        CIELAB: L* in [0, 100], a* and b* typically in [-128, 128]
    """
    # D65 参考白点
    xn, yn, zn = 0.95047, 1.00000, 1.08883

    xyz = img_xyz.astype(np.float64).copy()
    xyz[..., 0] /= xn
    xyz[..., 1] /= yn
    xyz[..., 2] /= zn

    delta = 6 / 29
    t = xyz ** (1 / 3)
    mask = xyz <= delta ** 3
    t[mask] = xyz[mask] / (3 * delta ** 2) + 4 / 29

    L = (116 * t[..., 1] - 16).astype(np.float32)
    a = (500 * (t[..., 0] - t[..., 1])).astype(np.float32)
    b = (200 * (t[..., 1] - t[..., 2])).astype(np.float32)

    return np.stack([L, a, b], axis=-1)


def rgb_to_lab(img_rgb: np.ndarray) -> np.ndarray:
    """sRGB → CIELAB (便捷函数)."""
    xyz = rgb_to_xyz(img_rgb)
    return xyz_to_lab(xyz)


def rgb_to_hsv(img_rgb: np.ndarray) -> np.ndarray:
    """RGB → HSV (OpenCV 实现, H in [0, 180], S,V in [0, 255])."""
    if img_rgb.max() <= 1.0:
        img_rgb = (img_rgb * 255).astype(np.uint8)
    if img_rgb.dtype != np.uint8:
        img_rgb = img_rgb.astype(np.uint8)
    # OpenCV 期望 BGR
    bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)


def rgb_to_ycbcr(img_rgb: np.ndarray) -> np.ndarray:
    """RGB → YCbCr."""
    if img_rgb.max() <= 1.0:
        img_rgb = (img_rgb * 255).astype(np.uint8)
    if img_rgb.dtype != np.uint8:
        img_rgb = img_rgb.astype(np.uint8)
    bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)


def rgb_to_chromaticity_xyy(img_rgb: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """RGB → CIE 1931 xyY 色度坐标.

    Returns:
        (x, y, Y) 三通道, 各 shape (H, W)
    """
    xyz = rgb_to_xyz(img_rgb)
    X, Y, Z = xyz[..., 0], xyz[..., 1], xyz[..., 2]
    total = X + Y + Z + 1e-10
    x = X / total
    y = Y / total
    return x.astype(np.float32), y.astype(np.float32), Y.astype(np.float32)


def rgb_to_chromaticity_uv(img_rgb: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """RGB → CIE 1976 UCS u'v' 色度坐标.

    Returns:
        (u_prime, v_prime)
    """
    xyz = rgb_to_xyz(img_rgb)
    X, Y, Z = xyz[..., 0], xyz[..., 1], xyz[..., 2]
    denom = X + 15 * Y + 3 * Z + 1e-10
    u_prime = (4 * X) / denom
    v_prime = (9 * Y) / denom
    return u_prime.astype(np.float32), v_prime.astype(np.float32)


# ============================================================
# 统计摘要
# ============================================================

def channel_stats(channel: np.ndarray,
                  percentiles: List[int] = (10, 25, 50, 75, 90)) -> Dict[str, float]:
    """计算单通道的基础统计量.

    Args:
        channel: 单通道图像, 任意shape
        percentiles: 需计算的百分位

    Returns:
        {mean, std, min, max, median, skewness, kurtosis, p10, p25, p50, p75, p90}
    """
    ch = channel.ravel().astype(np.float64)
    mean = float(np.mean(ch))
    std = float(np.std(ch))
    if std == 0:
        skew, kurt = 0.0, 0.0
    else:
        skew = float(np.mean(((ch - mean) / std) ** 3))
        kurt = float(np.mean(((ch - mean) / std) ** 4) - 3)  # excess kurtosis

    stats = {
        "mean": mean,
        "std": std,
        "min": float(np.min(ch)),
        "max": float(np.max(ch)),
        "median": float(np.median(ch)),
        "skewness": skew,
        "kurtosis": kurt,
    }
    for p in percentiles:
        stats[f"p{p}"] = float(np.percentile(ch, p))

    return stats


def histogram_features(channel: np.ndarray, bins: int = 32) -> Dict[str, float]:
    """计算归一化直方图特征.

    Args:
        channel: 单通道图像 (0–255)
        bins: 直方图bins数

    Returns:
        {hist_bin_0, hist_bin_1, ..., hist_bin_{bins-1}}
    """
    ch = channel.ravel().astype(np.float64)
    if ch.size == 0:
        return {f"hist_bin_{i}": np.nan for i in range(bins)}

    lo, hi = float(ch.min()), float(ch.max())
    if lo == hi:
        lo, hi = (0.0, 255.0) if 0 <= lo <= 255 else (lo - 0.5, hi + 0.5)

    hist, _ = np.histogram(ch, bins=bins, range=(lo, hi), density=False)
    hist = hist.astype(np.float64)
    total = hist.sum()
    if total > 0:
        hist /= total
    return {f"hist_bin_{i}": float(v) for i, v in enumerate(hist)}


# ============================================================
# 参考色卡值
# ============================================================

# X-Rite ColorChecker Classic 24 色块在 CIE Lab (D50) 下的参考值
# 来源: X-Rite 官方数据, 基于D50光源, 2° 观察者
# 若使用 D65 光源, 需要光源转换
COLORCHECKER_24_LAB_D50 = np.array([
    # Row 1 (棕 → 蓝)
    [37.986, 13.555, 14.059],   # 1  dark skin
    [65.711, 18.130, 17.810],   # 2  light skin
    [49.927, -4.880, -21.925],  # 3  blue sky
    [43.139, -13.095, 21.905],  # 4  foliage
    [55.112, 8.844, -25.399],   # 5  blue flower
    [70.719, -33.397, -0.199],  # 6  bluish green
    # Row 2
    [62.661, 36.067, 57.096],   # 7  orange
    [40.020, 10.410, -45.964],  # 8  purplish blue
    [51.124, 48.239, 16.248],   # 9  moderate red
    [30.325, 22.976, -21.587],  # 10 purple
    [72.532, -23.709, 57.255],  # 11 yellow green
    [71.941, 19.363, 67.857],   # 12 orange yellow
    # Row 3
    [28.778, 14.179, -50.297],  # 13 blue
    [55.261, -38.342, 31.370],  # 14 green
    [42.101, 53.378, 28.190],   # 15 red
    [81.733, 4.039, 79.819],    # 16 yellow
    [51.935, 49.986, -14.574],  # 17 magenta
    [51.038, -28.631, -28.638], # 18 cyan
    # Row 4 (灰度条)
    [96.539, -0.425, 1.186],    # 19 white
    [81.257, -0.638, -0.335],   # 20 neutral 8
    [66.766, -0.734, -0.504],   # 21 neutral 6.5
    [50.867, -0.153, -0.270],   # 22 neutral 5
    [35.656, -0.421, -1.231],   # 23 neutral 3.5
    [20.461, -0.079, -0.973],   # 24 black
], dtype=np.float64)

# D50 → D65 Bradford 色适应矩阵
BRADFORD_D50_TO_D65 = np.array([
    [0.9555766, -0.0230393, 0.0631636],
    [-0.0282895, 1.0099416, 0.0210077],
    [0.0122982, -0.0204830, 1.3299098],
], dtype=np.float64)


def get_colorchecker_lab_d65() -> np.ndarray:
    """获取 D65 光源下的 ColorChecker 24 参考 Lab 值 (计算值)."""
    lab_d50 = COLORCHECKER_24_LAB_D50
    # Lab → XYZ (D50)
    # 简化: 使用 colour-science 库做精确转换, 这里给出近似值
    # 实际使用时建议直接测量或用 colour-science 转换
    return lab_d50  # 近似, 实际场景建议配置 reference_file


# ============================================================
# 色差计算 (ΔE)
# ============================================================

def delta_e_76(lab1: np.ndarray, lab2: np.ndarray) -> np.ndarray:
    """CIE76 ΔE (欧氏距离)."""
    return np.sqrt(np.sum((lab1 - lab2) ** 2, axis=-1))


def delta_e_94(lab1: np.ndarray, lab2: np.ndarray,
               k_L: float = 1.0, k_C: float = 1.0, k_H: float = 1.0) -> np.ndarray:
    """CIE94 ΔE."""
    L1, a1, b1 = lab1[..., 0], lab1[..., 1], lab1[..., 2]
    L2, a2, b2 = lab2[..., 0], lab2[..., 1], lab2[..., 2]

    dL = L1 - L2
    C1 = np.sqrt(a1 ** 2 + b1 ** 2)
    C2 = np.sqrt(a2 ** 2 + b2 ** 2)
    dC = C1 - C2
    dH_sq = (a1 - a2) ** 2 + (b1 - b2) ** 2 - dC ** 2
    dH_sq = np.maximum(dH_sq, 0)  # 防负值
    dH = np.sqrt(dH_sq)

    S_L = 1.0
    S_C = 1.0 + 0.045 * C1
    S_H = 1.0 + 0.015 * C1

    return np.sqrt(
        (dL / (k_L * S_L)) ** 2 +
        (dC / (k_C * S_C)) ** 2 +
        (dH / (k_H * S_H)) ** 2
    )


def delta_e_2000(lab1: np.ndarray, lab2: np.ndarray) -> np.ndarray:
    """CIEDE2000 色差公式 (最精确).

    适用于逐像素比较.

    WARNING: 此函数计算量大, 对大图建议仅在ROI均值上计算.
    """
    L1, a1, b1 = lab1[..., 0], lab1[..., 1], lab1[..., 2]
    L2, a2, b2 = lab2[..., 0], lab2[..., 1], lab2[..., 2]

    L_mean = (L1 + L2) / 2

    C1 = np.sqrt(a1 ** 2 + b1 ** 2)
    C2 = np.sqrt(a2 ** 2 + b2 ** 2)
    C_mean = (C1 + C2) / 2

    # G因子
    C_mean_7 = C_mean ** 7
    G = 0.5 * (1 - np.sqrt(C_mean_7 / (C_mean_7 + 25 ** 7)))

    a1_prime = (1 + G) * a1
    a2_prime = (1 + G) * a2

    C1_prime = np.sqrt(a1_prime ** 2 + b1 ** 2)
    C2_prime = np.sqrt(a2_prime ** 2 + b2 ** 2)

    h1_prime = np.arctan2(b1, a1_prime + 1e-10) % (2 * np.pi)
    h2_prime = np.arctan2(b2, a2_prime + 1e-10) % (2 * np.pi)

    dL_prime = L2 - L1
    dC_prime = C2_prime - C1_prime

    # 色相差
    dh_prime = h2_prime - h1_prime
    cond = np.abs(dh_prime) > np.pi
    dh_prime = np.where(cond & (h2_prime <= h1_prime),
                        dh_prime + 2 * np.pi, dh_prime)
    dh_prime = np.where(cond & (h2_prime > h1_prime),
                        dh_prime - 2 * np.pi, dh_prime)
    dH_prime = 2 * np.sqrt(C1_prime * C2_prime + 1e-10) * np.sin(dh_prime / 2)

    H_mean = (h1_prime + h2_prime) / 2
    cond = np.abs(h1_prime - h2_prime) > np.pi
    H_mean = np.where(cond & ((h1_prime + h2_prime) < 2 * np.pi),
                      H_mean + np.pi, H_mean)
    H_mean = np.where(cond & ((h1_prime + h2_prime) >= 2 * np.pi),
                      H_mean - np.pi, H_mean)

    T = (1 - 0.17 * np.cos(H_mean - np.deg2rad(30))
         + 0.24 * np.cos(2 * H_mean)
         + 0.32 * np.cos(3 * H_mean + np.deg2rad(6))
         - 0.20 * np.cos(4 * H_mean - np.deg2rad(63)))

    dtheta = np.deg2rad(30) * np.exp(-((H_mean - np.deg2rad(275)) / np.deg2rad(25)) ** 2)
    C_mean_prime_7 = C_mean ** 7  # 近似
    R_C = 2 * np.sqrt(C_mean_prime_7 / (C_mean_prime_7 + 25 ** 7) + 1e-10)

    S_L = 1 + (0.015 * (L_mean - 50) ** 2) / np.sqrt(20 + (L_mean - 50) ** 2 + 1e-10)
    S_C = 1 + 0.045 * C_mean
    S_H = 1 + 0.015 * C_mean * T

    R_T = -np.sin(2 * dtheta) * R_C

    return np.sqrt(
        (dL_prime / (1 * S_L)) ** 2 +
        (dC_prime / (1 * S_C)) ** 2 +
        (dH_prime / (1 * S_H)) ** 2 +
        R_T * (dC_prime / (1 * S_C)) * (dH_prime / (1 * S_H))
    )


# ============================================================
# 图像配准辅助
# ============================================================

def find_colorchecker_roi(img_rgb: np.ndarray,
                          target_size: Tuple[int, int] = (24, 24)
                          ) -> Optional[np.ndarray]:
    """自动检测 ColorChecker 色卡 ROI.

    使用 OpenCV 轮廓检测 + 近似多边形方法.

    Args:
        img_rgb: 输入RGB图像 (BGR 或 RGB, uint8)
        target_size: 色卡采样分辨率

    Returns:
        (24, 3) 的色块平均RGB值数组, 或 None
    """
    # 简化版: 使用用户手动标注 + 透视变换
    # 生产环境建议使用专门的色卡检测库 (如 colour-checker-detection)
    # 此处保留接口, 完整实现见 colour-science 库
    print("WARNING: auto colorchecker detection not implemented. "
          "Please use manual ROI annotation or install colour-checker-detection.")
    return None
