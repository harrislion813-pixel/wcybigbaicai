"""
叶片分割模块 — 支持多种分割方法:
    1. 超绿指数 (ExG) 阈值分割 (传统方法, 无需GPU)
    2. GrabCut 交互式/自动分割
    3. U-Net 深度学习分割 (最高精度)
    4. 基于 SAM 的零样本分割
"""
from pathlib import Path
from typing import Optional, Tuple, Dict
import numpy as np
import cv2


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def normalize_imagenet_rgb(img_rgb: np.ndarray) -> np.ndarray:
    """Normalize a float RGB [0,1] image with ImageNet channel statistics."""
    img = img_rgb.astype(np.float32)
    if img.max() > 1.0:
        img = img / 255.0
    return (img - IMAGENET_MEAN) / IMAGENET_STD


class BaseSegmenter:
    """叶片分割器基类."""

    def __init__(self, morph_kernel_size: int = 5, min_area_ratio: float = 0.002,
                 exclude_border_components: bool = False,
                 border_margin_ratio: float = 0.01):
        if not isinstance(morph_kernel_size, int) or morph_kernel_size <= 0:
            raise ValueError("morph_kernel_size must be a positive integer")
        if not 0 <= min_area_ratio < 1:
            raise ValueError("min_area_ratio must be in [0, 1)")
        if not 0 <= border_margin_ratio < 0.5:
            raise ValueError("border_margin_ratio must be in [0, 0.5)")
        self.morph_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (morph_kernel_size, morph_kernel_size))
        self.min_area_ratio = min_area_ratio
        self.exclude_border_components = exclude_border_components
        self.border_margin_ratio = border_margin_ratio

    def segment(self, img_rgb: np.ndarray, **kwargs) -> np.ndarray:
        raise NotImplementedError

    def postprocess(self, mask: np.ndarray) -> np.ndarray:
        """后处理: 形态学去噪 + 去除小连通域."""
        mask = (mask > 0).astype(np.uint8) * 255
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.morph_kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.morph_kernel)

        height, width = mask.shape[:2]
        total_area = height * width
        min_area = total_area * self.min_area_ratio
        margin_x = int(round(width * self.border_margin_ratio))
        margin_y = int(round(height * self.border_margin_ratio))

        cleaned = np.zeros_like(mask)
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        for contour in contours:
            if cv2.contourArea(contour) < min_area:
                continue

            if self.exclude_border_components:
                x, y, w, h = cv2.boundingRect(contour)
                touches_border_margin = (
                    x <= margin_x or y <= margin_y or
                    x + w >= width - margin_x or
                    y + h >= height - margin_y
                )
                if touches_border_margin:
                    continue

            cv2.drawContours(cleaned, [contour], -1, 255, thickness=cv2.FILLED)
        return cleaned


class ExGSegmenter(BaseSegmenter):
    """超绿指数 (Excess Green Index) 阈值分割.

    原理: ExG = 2*G - R - B
          绿色叶片区域 ExG > 0, 背景 (土壤/非植被) ExG < 0.
          加上 Otsu 自适应阈值获得更鲁棒的结果.
    """

    def __init__(self, exg_threshold: float = 0.15, use_otsu: bool = True,
                 morph_kernel_size: int = 5, min_area_ratio: float = 0.002,
                 exclude_border_components: bool = False,
                 border_margin_ratio: float = 0.01):
        super().__init__(morph_kernel_size, min_area_ratio,
                         exclude_border_components, border_margin_ratio)
        self.exg_threshold = exg_threshold
        self.use_otsu = use_otsu

    @staticmethod
    def compute_exg(img_rgb: np.ndarray) -> np.ndarray:
        """计算超绿指数图. 输入 (H,W,3) float32 [0,1]"""
        r, g, b = img_rgb[..., 0], img_rgb[..., 1], img_rgb[..., 2]
        # 归一化: 避免光照差异
        total = r + g + b + 1e-10
        r_n, g_n, b_n = r / total, g / total, b / total
        return (2 * g_n - r_n - b_n).astype(np.float32)

    @staticmethod
    def compute_exgr(img_rgb: np.ndarray) -> np.ndarray:
        """超绿-超红差值: ExGR = ExG - ExR, 对阴影更鲁棒."""
        r, g, b = img_rgb[..., 0], img_rgb[..., 1], img_rgb[..., 2]
        total = r + g + b + 1e-10
        r_n, g_n, b_n = r / total, g / total, b / total
        exg = 2 * g_n - r_n - b_n
        exr = 1.4 * r_n - g_n
        return (exg - exr).astype(np.float32)

    def segment(self, img_rgb: np.ndarray, **kwargs) -> np.ndarray:
        """ExG + Otsu / 固定阈值分割."""
        exg = self.compute_exgr(img_rgb)

        if self.use_otsu:
            exg_norm = cv2.normalize(exg, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            _, mask = cv2.threshold(exg_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        else:
            mask = (exg > self.exg_threshold).astype(np.uint8) * 255

        return self.postprocess(mask)


class GrabCutSegmenter(BaseSegmenter):
    """基于 GrabCut 的叶片分割.

    支持两种模式:
        - manual: 手动绘制矩形ROI
        - auto: 自动用ExG结果初始化GrabCut (推荐批处理)
    """

    def __init__(self, iterations: int = 5, morph_kernel_size: int = 5,
                 min_area_ratio: float = 0.002,
                 exclude_border_components: bool = False,
                 border_margin_ratio: float = 0.01,
                 exg_threshold: float = 0.15,
                 use_otsu: bool = True):
        super().__init__(morph_kernel_size, min_area_ratio,
                         exclude_border_components, border_margin_ratio)
        self.iterations = iterations
        self.exg_threshold = exg_threshold
        self.use_otsu = use_otsu

    def segment(self, img_rgb: np.ndarray,
                rect: Optional[Tuple[int, int, int, int]] = None,
                init_mask: Optional[np.ndarray] = None,
                **kwargs) -> np.ndarray:
        """GrabCut 分割.

        Args:
            img_rgb: (H,W,3) float32 [0,1] or uint8 [0,255]
            rect: (x, y, w, h) ROI矩形. 若为None则用全图.
            init_mask: 初始掩膜 (可选, 用于auto模式)

        Returns:
            二值掩膜
        """
        if img_rgb.dtype != np.uint8:
            img_uint8 = (img_rgb * 255).astype(np.uint8)
        else:
            img_uint8 = img_rgb

        h, w = img_uint8.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)

        if rect is None:
            # 默认: 全图矩形, 留10%边距
            margin = 0.05
            rect = (int(w * margin), int(h * margin),
                    int(w * (1 - 2 * margin)), int(h * (1 - 2 * margin)))

        bgd_model = np.zeros((1, 65), dtype=np.float64)
        fgd_model = np.zeros((1, 65), dtype=np.float64)

        if init_mask is not None:
            mask = init_mask.copy()
            mask[mask > 0] = cv2.GC_PR_FGD
            cv2.grabCut(img_uint8, mask, rect, bgd_model, fgd_model,
                       self.iterations, cv2.GC_INIT_WITH_MASK)
        else:
            cv2.grabCut(img_uint8, mask, rect, bgd_model, fgd_model,
                       self.iterations, cv2.GC_INIT_WITH_RECT)

        # 提取前景
        final_mask = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
        return self.postprocess(final_mask)

    def segment_auto(self, img_rgb: np.ndarray) -> np.ndarray:
        """自动模式: 先ExG粗分割, 再用GrabCut精修."""
        exg_seg = ExGSegmenter(
            exg_threshold=self.exg_threshold,
            use_otsu=self.use_otsu,
            morph_kernel_size=self.morph_kernel.shape[0],
            min_area_ratio=self.min_area_ratio,
            exclude_border_components=self.exclude_border_components,
            border_margin_ratio=self.border_margin_ratio,
        )
        proposal = exg_seg.segment(img_rgb)
        if not np.any(proposal > 0):
            return proposal
        # 膨胀初始掩膜
        init_mask = cv2.dilate(proposal, np.ones((7, 7), np.uint8), iterations=2)
        # GrabCut精修
        try:
            refined = self.segment(img_rgb, init_mask=init_mask)
        except cv2.error:
            return proposal
        if not np.any(refined > 0):
            return proposal

        proposal_contours, _ = cv2.findContours(
            proposal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        refined_contours, _ = cv2.findContours(
            refined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if len(refined_contours) < len(proposal_contours):
            return proposal
        return refined


class UNetSegmenter(BaseSegmenter):
    """基于U-Net的深度学习叶片分割器.

    支持加载预训练模型进行推理.
    """

    def __init__(self, model_path: str, backbone: str = "efficientnet-b3",
                 device: str = "cuda", morph_kernel_size: int = 5,
                 min_area_ratio: float = 0.005,
                 exclude_border_components: bool = False,
                 border_margin_ratio: float = 0.01):
        super().__init__(morph_kernel_size, min_area_ratio,
                         exclude_border_components, border_margin_ratio)
        self.model_path = model_path
        self.backbone = backbone
        self.device = device
        self._model = None

    def _load_model(self):
        """延迟加载模型."""
        if self._model is not None:
            return
        import torch
        if str(self.device).startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available; use device='cpu'")
        try:
            import segmentation_models_pytorch as smp
        except ImportError:
            raise ImportError("需要安装 segmentation_models_pytorch: pip install segmentation-models-pytorch")

        self._model = smp.Unet(
            encoder_name=self.backbone,
            encoder_weights=None,  # 使用自定义权重
            in_channels=3,
            classes=1,
        )
        checkpoint = torch.load(self.model_path, map_location=self.device)
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            checkpoint_backbone = checkpoint.get("backbone")
            if checkpoint_backbone and checkpoint_backbone != self.backbone:
                raise ValueError(
                    f"Checkpoint backbone '{checkpoint_backbone}' does not match "
                    f"configured backbone '{self.backbone}'"
                )
            state_dict = checkpoint["state_dict"]
            self.threshold = float(checkpoint.get("threshold", 0.5))
        else:
            state_dict = checkpoint
            self.threshold = 0.5
        self._model.load_state_dict(state_dict)
        self._model.to(self.device)
        self._model.eval()
        print(f"[UNetSegmenter] Model loaded from {self.model_path} on {self.device}")

    def segment(self, img_rgb: np.ndarray, **kwargs) -> np.ndarray:
        """U-Net 叶片分割.

        Args:
            img_rgb: (H,W,3) float32 [0,1] or uint8 [0,255]

        Returns:
            二值掩膜
        """
        import torch
        self._load_model()

        if img_rgb.dtype == np.uint8:
            img = img_rgb.astype(np.float32) / 255.0
        else:
            img = img_rgb.astype(np.float32)

        h_orig, w_orig = img.shape[:2]

        # Resize到32的倍数 (U-Net要求)
        new_h = ((h_orig + 31) // 32) * 32
        new_w = ((w_orig + 31) // 32) * 32
        img_resized = cv2.resize(img, (new_w, new_h))

        # 与训练阶段保持一致的 ImageNet 归一化。
        img_resized = normalize_imagenet_rgb(img_resized)

        # To tensor: (H,W,C) → (1,C,H,W)
        tensor = torch.from_numpy(img_resized).permute(2, 0, 1).unsqueeze(0).float()
        tensor = tensor.to(self.device)

        with torch.no_grad():
            pred = self._model(tensor)
            pred = torch.sigmoid(pred).squeeze().cpu().numpy()

        # Resize回原始尺寸
        pred_resized = cv2.resize(pred, (w_orig, h_orig))
        mask = (pred_resized > self.threshold).astype(np.uint8) * 255

        return self.postprocess(mask)


class SAMSegmenter(BaseSegmenter):
    """基于 SAM (Segment Anything Model) 的零样本叶片分割.

    适合无标注数据的场景, 但速度较慢.
    需要 pip install segment-anything
    """

    def __init__(self, sam_checkpoint: str, model_type: str = "vit_h",
                 device: str = "cuda", morph_kernel_size: int = 5,
                 min_area_ratio: float = 0.005,
                 exclude_border_components: bool = False,
                 border_margin_ratio: float = 0.01):
        super().__init__(morph_kernel_size, min_area_ratio,
                         exclude_border_components, border_margin_ratio)
        self.sam_checkpoint = sam_checkpoint
        self.model_type = model_type
        self.device = device
        self._predictor = None

    def _load_predictor(self):
        if self._predictor is not None:
            return
        try:
            from segment_anything import sam_model_registry, SamPredictor
        except ImportError:
            raise ImportError("需要安装 segment-anything: pip install git+https://github.com/facebookresearch/segment-anything.git")

        sam = sam_model_registry[self.model_type](checkpoint=self.sam_checkpoint)
        sam.to(device=self.device)
        self._predictor = SamPredictor(sam)
        print(f"[SAMSegmenter] SAM ({self.model_type}) loaded on {self.device}")

    def segment(self, img_rgb: np.ndarray,
                center_point: Optional[Tuple[int, int]] = None,
                **kwargs) -> np.ndarray:
        """SAM 分割 (基于中心点prompt).

        Args:
            img_rgb: (H,W,3) uint8 [0,255]
            center_point: 中心点坐标, None则使用图像中心

        Returns:
            二值掩膜
        """
        self._load_predictor()

        if img_rgb.dtype != np.uint8:
            img_uint8 = (img_rgb * 255).astype(np.uint8)
        else:
            img_uint8 = img_rgb.copy()

        self._predictor.set_image(img_uint8)

        if center_point is None:
            h, w = img_uint8.shape[:2]
            center_point = (w // 2, h // 2)

        input_point = np.array([center_point])
        input_label = np.array([1])  # 1 = foreground

        masks, scores, _ = self._predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=True,
        )

        # 选择得分最高的掩膜
        best_idx = np.argmax(scores)
        mask = (masks[best_idx] * 255).astype(np.uint8)

        return self.postprocess(mask)


# ============================================================
# 自动方法选择工厂
# ============================================================

class AutoSegmenter(BaseSegmenter):
    """Lightweight automatic segmenter: ExG proposal followed by GrabCut refinement."""

    def __init__(self, iterations: int = 5, morph_kernel_size: int = 5,
                 min_area_ratio: float = 0.002,
                 exclude_border_components: bool = False,
                 border_margin_ratio: float = 0.01,
                 exg_threshold: float = 0.15,
                 use_otsu: bool = True):
        super().__init__(morph_kernel_size, min_area_ratio,
                         exclude_border_components, border_margin_ratio)
        self.grabcut = GrabCutSegmenter(
            iterations=iterations,
            morph_kernel_size=morph_kernel_size,
            min_area_ratio=min_area_ratio,
            exclude_border_components=exclude_border_components,
            border_margin_ratio=border_margin_ratio,
            exg_threshold=exg_threshold,
            use_otsu=use_otsu,
        )
        self.exg = ExGSegmenter(
            exg_threshold=exg_threshold,
            use_otsu=use_otsu,
            morph_kernel_size=morph_kernel_size,
            min_area_ratio=min_area_ratio,
            exclude_border_components=exclude_border_components,
            border_margin_ratio=border_margin_ratio,
        )

    def segment(self, img_rgb: np.ndarray, **kwargs) -> np.ndarray:
        mask = self.grabcut.segment_auto(img_rgb)
        if np.any(mask > 0):
            return mask
        return self.exg.segment(img_rgb)


def create_segmenter(method: str = "auto", **kwargs) -> BaseSegmenter:
    """根据配置创建分割器.

    Args:
        method: "exg" | "grabcut" | "unet" | "sam" | "auto"
        **kwargs: 传递给具体分割器的参数

    Returns:
        BaseSegmenter 实例
    """
    method = str(method).lower()
    method_map = {
        "exg": ExGSegmenter,
        "exg_threshold": ExGSegmenter,
        "grabcut": GrabCutSegmenter,
        "unet": UNetSegmenter,
        "sam": SAMSegmenter,
        "auto": AutoSegmenter,
    }

    if method == "exg_threshold":
        kwargs["use_otsu"] = False

    if method == "auto":
        # 自动选择: 优先 U-Net, 其次 GrabCut-auto, 兜底 ExG
        if "model_path" in kwargs and Path(kwargs["model_path"]).exists():
            seg_cls = UNetSegmenter
        else:
            seg_cls = AutoSegmenter

    else:
        seg_cls = method_map.get(method)
    if seg_cls is None:
        raise ValueError(f"Unknown segmentation method: {method}. "
                         f"Choose from {list(method_map.keys())} + 'auto'")

    # Allow parameters that belong to another supported backend so one shared
    # config can switch methods, but fail on genuinely unknown/typoed keys.
    import inspect
    known_params = set()
    for candidate in set(method_map.values()):
        known_params.update(inspect.signature(candidate.__init__).parameters)
    known_params.discard("self")
    unknown_params = sorted(set(kwargs) - known_params)
    if unknown_params:
        raise ValueError(f"Unknown segmentation parameters: {unknown_params}")

    sig = inspect.signature(seg_cls.__init__)
    valid_params = set(sig.parameters.keys()) - {"self"}
    filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_params}

    return seg_cls(**filtered_kwargs)
