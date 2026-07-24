"""
植被指数提取 — 基于RGB图像的植被指数, 用于量化叶片绿度/健康状况。

文献支持的RGB植被指数 (不需要NIR波段):
    VARI  — Visible Atmospherically Resistant Index
    GLI   — Green Leaf Index
    ExG   — Excess Green
    ExR   — Excess Red
    ExGR  — ExG - ExR
    NGRDI — Normalized Green-Red Difference Index
    DGCI  — Dark Green Color Index
    CIVE  — Color Index of Vegetation Extraction
    MGRVI — Modified Green Red Vegetation Index
    RGBVI — RGB-based Vegetation Index
    NDI   — Normalized Difference Index
    VEG   — Vegetation Index
    COM   — Combination Index
"""
from typing import Dict, List, Optional
import numpy as np


class VegetationIndexExtractor:
    """基于RGB的植被指数计算器.

    输入统一为 float RGB [0,1]。比例型指数直接使用该范围；
    ExG/ExR/ExGR 在公式内部使用色度归一化值；CIVE 按原始
    8-bit RGB 公式计算，以保持公式系数与输入尺度一致。

    Usage:
        vie = VegetationIndexExtractor()
        indices = vie.compute(img_rgb, mask)
    """

    def __init__(self, indices: Optional[List[str]] = None):
        """
        Args:
            indices: 要计算的指数列表, None表示计算全部.
                     可选: VARI, GLI, ExG, ExR, ExGR, NGRDI, DGCI,
                           CIVE, MGRVI, RGBVI, NDI, VEG, COM
        """
        default_indices = [
            "VARI", "GLI", "ExG", "ExR", "ExGR", "NGRDI", "DGCI",
            "CIVE", "MGRVI", "RGBVI", "NDI", "VEG", "COM"
        ]
        self.indices = default_indices if indices is None else list(indices)
        self._compute_fn = {
            "VARI": self._vari,
            "GLI": self._gli,
            "ExG": self._exg,
            "ExR": self._exr,
            "ExGR": self._exgr,
            "NGRDI": self._ngrdi,
            "DGCI": self._dgci,
            "CIVE": self._cive,
            "MGRVI": self._mgrvi,
            "RGBVI": self._rgbvi,
            "NDI": self._ndi,
            "VEG": self._veg,
            "COM": self._com,
        }
        unknown = sorted(set(self.indices) - set(self._compute_fn))
        if unknown:
            raise ValueError(f"Unknown vegetation indices: {unknown}")

    def compute(self, img_rgb: np.ndarray,
                mask: Optional[np.ndarray] = None) -> Dict[str, float]:
        """计算所有植被指数.

        Args:
            img_rgb: (H,W,3) float32 [0,1] or uint8 [0,255]
            mask: (H,W) uint8 叶片掩膜, None = 全图

        Returns:
            {index_name: value} 字典
        """
        if img_rgb.dtype == np.uint8:
            img = img_rgb.astype(np.float32) / 255.0
        else:
            img = img_rgb.astype(np.float32)

        R, G, B = img[..., 0], img[..., 1], img[..., 2]
        if mask is not None:
            if mask.max() > 1:
                mask_bin = mask > 127
            else:
                mask_bin = mask > 0.5
            # Work only on leaf pixels. This avoids allocating one full-resolution
            # index map per formula when the leaf occupies a small part of the frame.
            R, G, B = R[mask_bin], G[mask_bin], B[mask_bin]
        else:
            R, G, B = R.ravel(), G.ravel(), B.ravel()

        results = {}
        for idx_name in self.indices:
            vals = self._compute_fn[idx_name](R, G, B)

            if vals.size == 0:
                results[idx_name] = np.nan
                continue

            finite = vals[np.isfinite(vals)]
            if finite.size == 0:
                results[idx_name] = np.nan
                results[f"{idx_name}_std"] = np.nan
                results[f"{idx_name}_median"] = np.nan
                continue

            results[idx_name] = float(finite.mean())
            results[f"{idx_name}_std"] = float(finite.std())
            results[f"{idx_name}_median"] = float(np.median(finite))

        return results

    # ----------------------------------------------------------
    # 各指数定义
    # ----------------------------------------------------------

    @staticmethod
    def _vari(R, G, B):
        """VARI: Visible Atmospherically Resistant Index.
        VARI = (G - R) / (G + R - B)
        范围: 通常 -1 ~ 1, 正值表示绿色植被
        文献: Gitelson et al. (2002), Journal of Plant Physiology
        """
        return (G - R) / (G + R - B + 1e-10)

    @staticmethod
    def _gli(R, G, B):
        """GLI: Green Leaf Index.
        GLI = (2*G - R - B) / (2*G + R + B)
        范围: -1 ~ 1, 正值=绿色
        与SPAD/叶绿素含量高度相关 (r=0.8-0.9)
        文献: Louhaichi et al. (2001)
        """
        return (2 * G - R - B) / (2 * G + R + B + 1e-10)

    @staticmethod
    def _exg(R, G, B):
        """ExG: Excess Green Index.
        ExG = 2*G - R - B  (归一化后)
        最经典的RGB植被指数, 对绿色植被区分力强
        文献: Woebbecke et al. (1995), TASAE
        """
        normalize = R + G + B + 1e-10
        return 2 * G / normalize - R / normalize - B / normalize

    @staticmethod
    def _exr(R, G, B):
        """ExR: Excess Red Index.
        ExR = 1.4*R - G
        文献: Meyer et al. (1999)
        """
        normalize = R + G + B + 1e-10
        return 1.4 * R / normalize - G / normalize

    @staticmethod
    def _exgr(R, G, B):
        """ExGR: ExG - ExR.
        对阴影和非绿色背景鲁棒性更好
        文献: Neto et al. (2006)
        """
        normalize = R + G + B + 1e-10
        g, r, b = G / normalize, R / normalize, B / normalize
        exg = 2 * g - r - b
        exr = 1.4 * r - g
        return exg - exr

    @staticmethod
    def _ngrdi(R, G, B):
        """NGRDI: Normalized Green-Red Difference Index.
        NGRDI = (G - R) / (G + R)
        与叶绿素正相关
        """
        return (G - R) / (G + R + 1e-10)

    @staticmethod
    def _dgci(R, G, B):
        """DGCI: Dark Green Color Index.
        DGCI = [(H - 60)/60 + (1 - S) + (1 - V)] / 3
        其中 HSV 归一化到 [0,1]
        专用于量化叶片暗绿色程度, 与叶绿素含量高度相关
        文献: Karcher & Richardson (2003), Crop Science
        """
        # 需要将RGB → HSV (用numpy实现, 避免OpenCV循环)
        Rc, Gc, Bc = R, G, B

        max_val = np.maximum(np.maximum(Rc, Gc), Bc)
        min_val = np.minimum(np.minimum(Rc, Gc), Bc)
        delta = max_val - min_val

        # 色相 H in [0, 360]
        H = np.zeros_like(Rc)
        mask_r = (max_val == Rc)
        mask_g = (max_val == Gc)
        mask_b = (max_val == Bc)

        H = np.where(mask_g & (delta > 0), 60 * ((Bc - Rc) / (delta + 1e-10)) + 120, H)
        H = np.where(mask_b & (delta > 0), 60 * ((Rc - Gc) / (delta + 1e-10)) + 240, H)
        H = np.where(mask_r & (delta > 0), 60 * (((Gc - Bc) / (delta + 1e-10)) % 6), H)
        H = H / 360.0  # → [0, 1]

        # 饱和度 S
        S = np.where(max_val > 0, delta / (max_val + 1e-10), 0)

        # 明度 V
        V = max_val

        DGCI = ((H - 60/360) / (60/360) + (1 - S) + (1 - V)) / 3.0
        return DGCI

    @staticmethod
    def _cive(R, G, B):
        """CIVE: Color Index of Vegetation Extraction.
        CIVE = 0.441*R8 - 0.811*G8 + 0.385*B8 + 18.78745,
        其中 R8/G8/B8 为 [0,255] 的 8-bit 等价值。
        文献: Kataoka et al. (2003)
        """
        return (0.441 * (R * 255.0) - 0.811 * (G * 255.0) +
                0.385 * (B * 255.0) + 18.78745)

    @staticmethod
    def _mgrvi(R, G, B):
        """MGRVI: Modified Green Red Vegetation Index.
        MGRVI = (G² - R²) / (G² + R²)
        文献: Bendig et al. (2015)
        """
        return (G**2 - R**2) / (G**2 + R**2 + 1e-10)

    @staticmethod
    def _rgbvi(R, G, B):
        """RGBVI: RGB-based Vegetation Index.
        RGBVI = (G² - R*B) / (G² + R*B)
        文献: Bendig et al. (2015)
        """
        return (G**2 - R * B) / (G**2 + R * B + 1e-10)

    @staticmethod
    def _ndi(R, G, B):
        """NDI: Normalized Difference Index.
        NDI = (G - R) / (G + R)  # 等价于 NGRDI
        """
        return (G - R) / (G + R + 1e-10)

    @staticmethod
    def _veg(R, G, B):
        """VEG: Vegetation index.
        VEG = G / (R^a * B^(1-a)), a = 0.667
        文献: Hague et al. (2006)
        """
        a = 0.667
        return G / (R**a * B**(1 - a) + 1e-10)

    @staticmethod
    def _com(R, G, B):
        """COM: Combination index.
        COM = ExG + CIVE + ExGR + VEG
        文献: Guerrero et al. (2012)
        """
        normalize = R + G + B + 1e-10
        g, r, b = G / normalize, R / normalize, B / normalize
        exg = 2 * g - r - b
        exr = 1.4 * r - g
        exgr = exg - exr
        cive = (0.441 * (R * 255.0) - 0.811 * (G * 255.0) +
                0.385 * (B * 255.0) + 18.78745)
        a = 0.667
        veg = G / (R**a * B**(1 - a) + 1e-10)
        return exg + cive + exgr + veg


# ============================================================
# SPAD/叶绿素含量估算模型
# ============================================================
def estimate_chlorophyll_from_rgb(img_rgb: np.ndarray,
                                  mask: np.ndarray = None,
                                  calibration: str = "liang_2015") -> Dict[str, float]:
    """利用RGB植被指数估算SPAD/叶绿素含量.

    基于已发表的RGB→SPAD校准模型.
    注意: 这些模型具有品种特异性和环境依赖性, 建议用少量SPAD实测值做本地校准.

    Args:
        img_rgb: RGB图像
        mask: 叶片掩膜
        calibration: 校准模型选择
            - "liang_2015": SPAD = 21.93 + 47.48 * GLI (水稻, R²=0.91)
            - "wang_2014": SPAD = 17.6 + 35.8 * VARI (小麦, R²=0.87)
            - "hunt_2013": Chlorophyll = 0.143 * DGCI + 0.003 (玉米, R²=0.82)

    Returns:
        {"estimated_SPAD": ..., "estimated_chlorophyll_mg_g": ...}
    """
    vie = VegetationIndexExtractor()
    indices = vie.compute(img_rgb, mask)

    results = {}
    if calibration == "liang_2015":
        gli = indices.get("GLI", 0)
        results["estimated_SPAD"] = 21.93 + 47.48 * gli
    elif calibration == "wang_2014":
        vari = indices.get("VARI", 0)
        results["estimated_SPAD"] = 17.6 + 35.8 * vari
    elif calibration == "hunt_2013":
        dgci = indices.get("DGCI", 0)
        results["estimated_chlorophyll_mg_g"] = 0.143 * dgci + 0.003

    return results
