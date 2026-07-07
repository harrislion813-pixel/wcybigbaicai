"""
主流水线编排器 — 串联 预处理 → 分割 → 特征提取 → 输出 的完整流程。

支持:
    - 单图处理: 快速验证
    - 批量处理: 用于大规模GWAS表型数据生成
    - 多重复合并: 按样本ID汇总多张图像 → 均值±SD
"""
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

import numpy as np
import cv2
import pandas as pd

from .preprocessing import ImagePreprocessor
from .segmentation import BaseSegmenter, ExGSegmenter, GrabCutSegmenter, create_segmenter
from .color_features import ColorFeatureExtractor
from .vegetation_indices import VegetationIndexExtractor
from .texture_features import (
    GLCMTextureExtractor, LeafShapeExtractor, ColorTextureAnalyzer
)
from .utils import find_images, parse_sample_id, safe_mkdir, write_image_rgb


_SEGMENTATION_ALIASES = {
    "min_leaf_area_ratio": "min_area_ratio",
    "grabcut_iterations": "iterations",
    "unet_model": "model_path",
    "model": "model_path",
}


class LeafColorPipeline:
    """大白菜叶色表型提取完整流水线.

    典型用法:
        pipeline = LeafColorPipeline(config_dict)
        pipeline.process_single("path/to/image.jpg")
        # 或
        pipeline.process_batch("data/raw_images/", output_csv="phenotypes.csv")

    输出表型表格式说明 (每行一个样本):
        +------------------+-----------------+------------------+
        | sample_id        | RGB_R_mean      | RGB_G_mean       | ...
        | rep              | HSV_H_mean      | CIELAB_L_mean    | ...
        | developmental_st | GLI             | VARI             | ...
        +------------------+-----------------+------------------+
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        Args:
            config: 配置字典 (可从 config.yaml 加载).
                    若为 None, 使用默认配置.
        """
        self.config = config or {}

        # ---- 初始化各模块 ----
        self.preprocessor = self._init_preprocessor()
        self.segmenter = self._init_segmenter()
        self.color_extractor = self._init_color_extractor()
        self.veg_index_extractor = self._init_veg_index_extractor()
        self.texture_extractor = self._init_texture_extractor()
        self.shape_extractor = self._init_shape_extractor()
        self.color_texture_analyzer = ColorTextureAnalyzer()

        # ---- 输出配置 ----
        output = self.config.get("output", {})
        self.output_format = output.get("format", "csv")

    def _init_preprocessor(self) -> ImagePreprocessor:
        calib = self.config.get("color_calibration", {})
        return ImagePreprocessor(
            calibration_method=calib.get("method", "polynomial"),
            polynomial_degree=calib.get("polynomial_degree", 2),
        )

    def _init_segmenter(self) -> BaseSegmenter:
        seg = self.config.get("segmentation", {})
        method = seg.get("method", "exg")
        kwargs = {}
        for key, value in seg.items():
            if key == "method":
                continue
            kwargs[_SEGMENTATION_ALIASES.get(key, key)] = value
        return create_segmenter(method, **kwargs)

    def _init_color_extractor(self) -> ColorFeatureExtractor:
        ft = self.config.get("features", {})
        color_spaces = ft.get("color_spaces", ["RGB", "HSV", "CIELAB", "YCbCr"])
        hist_cfg = ft.get("histogram", {})
        return ColorFeatureExtractor(
            color_spaces=color_spaces,
            hist_bins=hist_cfg.get("bins", 32),
            hist_percentiles=hist_cfg.get("percentiles", [10, 25, 50, 75, 90]),
            include_color_moments=ft.get("color_moments", True),
            include_histogram=hist_cfg.get("enabled", True),
            include_chromaticity=ft.get("chromaticity", {}).get("enabled", True),
        )

    def _init_veg_index_extractor(self) -> VegetationIndexExtractor:
        ft = self.config.get("features", {})
        vi = ft.get("vegetation_indices", {})
        if vi.get("enabled", True):
            indices = vi.get("indices", None)  # None = all
        else:
            indices = []
        return VegetationIndexExtractor(indices=indices)

    def _init_texture_extractor(self) -> GLCMTextureExtractor:
        ft = self.config.get("features", {})
        tex = ft.get("texture", {})
        if tex.get("enabled", True):
            return GLCMTextureExtractor(
                distances=tex.get("distances", [1, 3, 5]),
                angles=tex.get("angles", [0, 45, 90, 135]),
                levels=tex.get("levels", 64),
                properties=tex.get("properties", None),
            )
        return GLCMTextureExtractor(properties=[])

    def _init_shape_extractor(self) -> LeafShapeExtractor:
        ft = self.config.get("features", {})
        shape = ft.get("shape", {})
        if shape.get("enabled", True):
            return LeafShapeExtractor(features=shape.get("features", None))
        return LeafShapeExtractor(features=[])

    # ----------------------------------------------------------
    # 单张图像处理
    # ----------------------------------------------------------
    def process_single(self,
                       image_path: str,
                       sample_id: Optional[str] = None,
                       replicate: Optional[str] = None,
                       developmental_stage: Optional[str] = None,
                       metadata: Optional[Dict] = None,
                       white_balance: str = "gray_world",
                       return_visualization: bool = False
                       ) -> Dict[str, object]:
        """处理单张叶片图像, 提取完整叶色表型.

        处理流程:
            1. 读取图像 (支持 RAW/PNG/JPG/TIFF)
            2. 颜色校准 (可选, 需要预先计算CCM)
            3. 白平衡
            4. 叶片分割
            5. 多颜色空间特征提取
            6. 植被指数计算
            7. 纹理特征提取
            8. 形状特征提取
            9. 均匀性分析

        Args:
            image_path: 图像路径
            sample_id: 样本ID, None则从文件名提取
            replicate: 生物学重复编号
            developmental_stage: 发育时期 (苗期/莲座期/结球期)
            metadata: 额外元数据字典 (如种植地块、处理条件等)
            white_balance: 白平衡方法
            return_visualization: 是否返回可视化图像

        Returns:
            {
                "sample_id": str,
                "features": Dict[str, float],  # 所有表型特征
                "mask": np.ndarray | None,      # 分割掩膜 (可选)
                "visualization": np.ndarray | None  # 可视化图 (可选)
            }
        """
        t_start = time.time()

        # ---- Step 1: 预处理 ----
        img_path = Path(image_path)
        preprocessed = self.preprocessor.process(
            str(img_path), white_balance_method=white_balance
        )
        img_rgb = preprocessed["rgb"]
        img_uint8 = preprocessed["rgb_uint8"]
        img_lab = preprocessed["lab"]

        # ---- Step 2: 分割 ----
        mask = self.segmenter.segment(img_rgb)
        mask_qc = self._mask_qc(mask)

        # ---- Step 3: 颜色特征 ----
        color_feats = self.color_extractor.extract(img_uint8, mask)

        # ---- Step 4: 植被指数 ----
        veg_feats = self.veg_index_extractor.compute(img_rgb, mask)

        # ---- Step 5: 纹理特征 ----
        texture_feats = self.texture_extractor.compute(img_uint8, mask)

        # ---- Step 6: 形状特征 ----
        shape_feats = self.shape_extractor.compute(mask)

        # ---- Step 7: 颜色均匀性 ----
        uniformity_feats = self.color_texture_analyzer.color_uniformity(img_lab, mask)

        # ---- 汇总所有特征 ----
        all_features: Dict[str, float] = {}
        all_features.update(color_feats)
        all_features.update(veg_feats)
        all_features.update(texture_feats)
        all_features.update(shape_feats)
        all_features.update(uniformity_feats)
        all_features.update(mask_qc)

        # ---- 元数据 ----
        sid = sample_id or parse_sample_id(img_path.name)
        result: Dict[str, object] = {
            "sample_id": sid,
            "features": all_features,
            "image_path": str(img_path),
        }

        if replicate is not None:
            result["replicate"] = replicate
        if developmental_stage is not None:
            result["developmental_stage"] = developmental_stage
        if metadata:
            result["metadata"] = metadata

        # ---- 可视化 (可选) ----
        if return_visualization:
            result["mask"] = mask
            result["visualization"] = self._create_visualization(
                img_uint8, mask, all_features
            )

        elapsed = time.time() - t_start
        print(f"  [{sid}] Processed in {elapsed:.2f}s → {len(all_features)} features extracted")

        return result

    # ----------------------------------------------------------
    # 批量处理
    # ----------------------------------------------------------
    def process_batch(self,
                      image_dir: str,
                      output_dir: Optional[str] = None,
                      output_csv: Optional[str] = None,
                      id_pattern: Optional[str] = None,
                      group_by_sample: bool = True,
                      white_balance: str = "gray_world",
                      save_visualizations: bool = False,
                      visualization_dir: Optional[str] = None,
                      verbose: bool = True) -> pd.DataFrame:
        """批量处理目录下的所有图像.

        Args:
            image_dir: 图像目录
            output_dir: 结果输出目录 (可选)
            output_csv: 输出CSV文件名 (含路径)
            id_pattern: 样本ID提取正则模式, None=自动
            group_by_sample: 是否按样本ID汇总多张图像
            white_balance: 白平衡方法
            verbose: 是否打印进度

        Returns:
            DataFrame: 表型表
        """
        img_paths = find_images(image_dir)
        if verbose:
            print(f"Found {len(img_paths)} images in {image_dir}")

        if not img_paths:
            print("WARNING: No images found!")
            return pd.DataFrame()

        vis_dir = None
        if save_visualizations:
            if visualization_dir:
                vis_dir = Path(visualization_dir)
            elif output_csv:
                vis_dir = Path(output_csv).parent / "visualizations"
            else:
                vis_dir = Path(output_dir or ".") / "visualizations"
            safe_mkdir(vis_dir)

        # 逐张处理
        records = []
        for i, img_path in enumerate(img_paths):
            if verbose:
                print(f"[{i+1}/{len(img_paths)}] Processing {img_path.name}...")

            try:
                sid = parse_sample_id(img_path.name, id_pattern) if id_pattern else None
                result = self.process_single(
                    str(img_path),
                    sample_id=sid,
                    white_balance=white_balance,
                    return_visualization=save_visualizations,
                )
                if save_visualizations and vis_dir is not None and "visualization" in result:
                    write_image_rgb(vis_dir / f"{img_path.stem}_vis.png", result["visualization"])
                    result.pop("visualization", None)
                    result.pop("mask", None)
                records.append(result)
            except Exception as e:
                print(f"  ERROR processing {img_path.name}: {e}")
                continue

        if not records:
            return pd.DataFrame()

        # 构建DataFrame
        rows = []
        for rec in records:
            row = {"sample_id": rec["sample_id"],
                   "image_path": rec["image_path"]}
            row.update(rec.get("metadata") or {})
            row.update(rec["features"])
            rows.append(row)

        df = pd.DataFrame(rows)

        # 按样本ID汇总 (多重复/)
        if group_by_sample:
            df = self._aggregate_by_sample(df)

        # 保存
        if output_csv or output_dir:
            csv_path = output_csv or str(
                Path(output_dir or ".") / "leaf_color_phenotypes.csv"
            )
            safe_mkdir(Path(csv_path).parent)
            df.to_csv(csv_path, index=False)
            print(f"\nPhenotype table saved to: {csv_path}")
            print(f"  Shape: {df.shape[0]} samples × {df.shape[1]} traits")

        return df

    # ----------------------------------------------------------
    # 按样本汇总
    # ----------------------------------------------------------
    @staticmethod
    def _mask_qc(mask: np.ndarray) -> Dict[str, float]:
        """Summarize segmentation mask quality for downstream filtering."""
        if mask.max() > 1:
            mask_bin = mask > 127
        else:
            mask_bin = mask > 0

        area_px = int(mask_bin.sum())
        total_px = int(mask_bin.size)
        feats = {
            "QC_mask_area_px": float(area_px),
            "QC_mask_area_ratio": float(area_px / total_px) if total_px else np.nan,
            "QC_mask_is_empty": float(area_px == 0),
        }

        if area_px == 0:
            feats.update({
                "QC_bbox_x": np.nan,
                "QC_bbox_y": np.nan,
                "QC_bbox_width": np.nan,
                "QC_bbox_height": np.nan,
            })
            return feats

        ys, xs = np.where(mask_bin)
        feats.update({
            "QC_bbox_x": float(xs.min()),
            "QC_bbox_y": float(ys.min()),
            "QC_bbox_width": float(xs.max() - xs.min() + 1),
            "QC_bbox_height": float(ys.max() - ys.min() + 1),
        })
        return feats

    @staticmethod
    def _aggregate_by_sample(df: pd.DataFrame,
                             trait_columns: Optional[List[str]] = None
                             ) -> pd.DataFrame:
        """按样本ID汇总多重复测量.

        对数值型特征计算: mean, std, cv (变异系数)
        对非数值列: 取第一个值
        """
        if "sample_id" not in df.columns:
            return df

        if trait_columns is None:
            # 自动识别数值列
            trait_columns = [
                c for c in df.columns
                if c not in ("sample_id", "image_path", "replicate",
                             "developmental_stage")
                and pd.api.types.is_numeric_dtype(df[c])
            ]

        if not trait_columns:
            return df.groupby("sample_id", as_index=False).first()

        # 分组聚合
        agg_dict = {}
        for col in trait_columns:
            agg_dict[col] = ["mean", "std"]

        grouped = df.groupby("sample_id").agg(agg_dict)

        # 扁平化列名. Keep per-image feature names intact; add rep_* for replicate stats
        grouped.columns = [
            col[0] if col[1] == "mean" else f"{col[0]}_rep_{col[1]}"
            for col in grouped.columns
        ]

        # 添加变异系数 (CV)
        cv_columns = {}
        for col in trait_columns:
            mean_col = col
            std_col = f"{col}_rep_std"
            if mean_col in grouped.columns and std_col in grouped.columns:
                cv_columns[f"{col}_rep_cv"] = (
                    grouped[std_col] / (grouped[mean_col].abs() + 1e-10)
                )
        if cv_columns:
            grouped = pd.concat([grouped, pd.DataFrame(cv_columns, index=grouped.index)], axis=1)

        # 添加重复数
        n_replicates = df.groupby("sample_id").size().rename("n_replicates")
        grouped = pd.concat([grouped, n_replicates], axis=1).copy()

        return grouped.reset_index()

    # ----------------------------------------------------------
    # 可视化
    # ----------------------------------------------------------
    @staticmethod
    def _create_visualization(img_uint8: np.ndarray,
                              mask: np.ndarray,
                              features: Dict[str, float]
                              ) -> np.ndarray:
        """生成叶色分析可视化图像.

        包含:
            - 原图 + 分割轮廓
            - CIELAB a*热力图 (反映红绿分布)
            - 关键特征文本标注
        """
        h, w = img_uint8.shape[:2]

        # 轮廓叠加
        vis = img_uint8.copy()
        if mask.max() > 1:
            mask_bin = mask > 127
        else:
            mask_bin = mask > 0
        contours, _ = cv2.findContours(
            mask_bin.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(vis, contours, -1, (0, 255, 0), max(2, w // 400))

        # 信息叠加
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = max(0.35, w / 1200)
        y = max(20, int(30 * scale))
        outline_thickness = max(1, int(scale * 3))
        text_thickness = max(1, int(scale * 2))
        line_step = max(12, int(25 * scale))
        key_items = [
            ("L*", features.get("CIELAB_L_mean", 0), "{:.1f}"),
            ("a*", features.get("CIELAB_A_mean", 0), "{:.2f}"),
            ("b*", features.get("CIELAB_B_mean", 0), "{:.2f}"),
            ("GLI", features.get("GLI", 0), "{:.3f}"),
            ("DGCI", features.get("DGCI", 0), "{:.3f}"),
            ("Area", features.get("Shape_area", 0), "{:.0f}"),
        ]
        for label, val, fmt in key_items:
            if np.isnan(val):
                continue
            text = f"{label}: {fmt.format(val)}"
            cv2.putText(vis, text, (10, y), font, scale, (255, 255, 255), outline_thickness)
            cv2.putText(vis, text, (10, y), font, scale, (0, 0, 0), text_thickness)
            y += line_step

        return vis
