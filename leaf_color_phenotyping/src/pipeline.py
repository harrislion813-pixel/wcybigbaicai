"""
主流水线编排器 — 串联 预处理 → 分割 → 特征提取 → 输出 的完整流程。

支持:
    - 单图处理: 快速验证
    - 批量处理: 用于大规模GWAS表型数据生成
    - 多重复合并: 按样本ID汇总多张图像 → 均值±SD
"""
import hashlib
import json
import platform
import sys
import time
from copy import deepcopy
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
import cv2
import pandas as pd

from .preprocessing import ImagePreprocessor
from .color_calibration import (
    CalibrationProfile,
    apply_calibration_profile,
    load_calibration_profile,
)
from .segmentation import BaseSegmenter, ExGSegmenter, GrabCutSegmenter, create_segmenter
from .color_features import ColorFeatureExtractor
from .vegetation_indices import VegetationIndexExtractor
from .texture_features import (
    GLCMTextureExtractor, LeafShapeExtractor, ColorTextureAnalyzer
)
from .utils import (
    RAW_IMAGE_EXTENSIONS, find_images, parse_sample_id, rgb_to_hsv, rgb_to_lab, safe_mkdir,
    write_image_rgb,
)


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
        self._validate_config(self.config)

        imaging = self.config.get("imaging", {})
        calibration = self.config.get("color_calibration", {})
        self.color_calibration_mode = self._resolve_color_calibration_mode(calibration)
        self.color_calibration_profile: Optional[CalibrationProfile] = None
        self.color_calibration_status = (
            "off" if self.color_calibration_mode == "off" else "not_configured"
        )
        self.color_calibration_source: Optional[str] = None
        self.color_calibration_matrix_sha256: Optional[str] = None
        self.default_white_balance = imaging.get("white_balance", "none")
        self.camera_id = imaging.get("camera_id", "")
        self.default_exposure_normalization = imaging.get(
            "exposure_normalization", "fixed_capture"
        )
        gray_card_rgb = imaging.get("gray_card_rgb")
        self.default_gray_roi = (
            np.asarray(gray_card_rgb, dtype=np.float32)
            if gray_card_rgb is not None else None
        )

        segmentation = self.config.get("segmentation", {})
        self.component_policy = segmentation.get("component_policy", "largest")
        self.component_min_exg = float(segmentation.get("component_min_exg", 0.30))
        self.max_segmentation_dimension = segmentation.get(
            "max_processing_dimension", 2200
        )
        self.normalize_segmentation_illumination = segmentation.get(
            "normalize_illumination", True
        )
        self.exclude_white_tissue = segmentation.get(
            "exclude_white_tissue", True
        )
        self.white_tissue_max_saturation = float(
            segmentation.get("white_tissue_max_saturation", 0.25)
        )
        self.white_tissue_min_retained_fraction = float(
            segmentation.get("white_tissue_min_retained_fraction", 0.50)
        )

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
        self.output_format = str(output.get("format", "csv")).lower()
        if self.output_format not in {"csv", "excel", "json"}:
            raise ValueError("output.format must be one of: csv, excel, json")
        self.output_table_name = output.get("phenotype_table_name", "leaf_color_phenotypes")
        self.default_save_visualizations = output.get("separate_visualization", False)
        self.save_raw_table = output.get("save_raw_table", True)
        self.aggregate_cv = output.get("aggregate_cv", False)
        self.write_manifest = output.get("write_manifest", True)
        self.last_batch_failures: List[Dict[str, str]] = []
        self.last_batch_cancelled = False

    @staticmethod
    def _validate_config(config: Dict) -> None:
        """Validate high-impact configuration fields before any image is processed."""
        if not isinstance(config, dict):
            raise TypeError("config must be a dictionary")

        imaging = config.get("imaging", {})
        if not isinstance(imaging, dict):
            raise TypeError("imaging config must be a mapping")
        white_balance = imaging.get("white_balance", "none")
        if white_balance not in {"gray_world", "perfect_reflector", "gray_card", "none"}:
            raise ValueError(f"Unknown white balance method: {white_balance}")
        if imaging.get("bits_per_channel", 16) not in (8, 16):
            raise ValueError("imaging.bits_per_channel must be 8 or 16")
        if not isinstance(imaging.get("raw_use_camera_wb", True), bool):
            raise TypeError("imaging.raw_use_camera_wb must be boolean")
        if not isinstance(
            imaging.get("exposure_normalization", "fixed_capture"), str
        ):
            raise TypeError("imaging.exposure_normalization must be a string")
        if not isinstance(imaging.get("camera_id", ""), str):
            raise TypeError("imaging.camera_id must be a string")
        gray_card_rgb = imaging.get("gray_card_rgb")
        if white_balance == "gray_card" and gray_card_rgb is None:
            raise ValueError("imaging.gray_card_rgb is required for gray_card white balance")
        if gray_card_rgb is not None:
            try:
                gray_card = np.asarray(gray_card_rgb, dtype=np.float64)
            except (TypeError, ValueError) as exc:
                raise ValueError("imaging.gray_card_rgb must contain three numbers") from exc
            if (
                gray_card.shape != (3,)
                or not np.isfinite(gray_card).all()
                or np.any(gray_card <= 0)
                or np.any(gray_card > 1)
            ):
                raise ValueError("imaging.gray_card_rgb must contain three values in (0, 1]")

        calibration = config.get("color_calibration", {})
        if not isinstance(calibration, dict):
            raise TypeError("color_calibration config must be a mapping")
        if "mode" in calibration and calibration["mode"] not in {
            "off", "optional", "required",
        }:
            raise ValueError(
                "color_calibration.mode must be 'off', 'optional', or 'required'"
            )
        for key in ("enabled", "allow_legacy_matrix"):
            if key in calibration and not isinstance(calibration[key], bool):
                raise TypeError(f"color_calibration.{key} must be boolean")
        for key in ("profile_file", "ccm_file"):
            if key in calibration and not isinstance(calibration[key], str):
                raise TypeError(f"color_calibration.{key} must be a string")

        segmentation = config.get("segmentation", {})
        if not isinstance(segmentation, dict):
            raise TypeError("segmentation config must be a mapping")
        if segmentation.get("component_policy", "largest") not in {"largest", "all"}:
            raise ValueError("segmentation.component_policy must be 'largest' or 'all'")
        component_min_exg = segmentation.get("component_min_exg", 0.30)
        if not isinstance(component_min_exg, (int, float)) or not -1 <= component_min_exg <= 2:
            raise ValueError("segmentation.component_min_exg must be in [-1, 2]")
        max_dimension = segmentation.get("max_processing_dimension", 2200)
        if max_dimension is not None and (
            not isinstance(max_dimension, int) or max_dimension < 256
        ):
            raise ValueError(
                "segmentation.max_processing_dimension must be null or an integer >= 256"
            )
        if not isinstance(segmentation.get("normalize_illumination", True), bool):
            raise TypeError("segmentation.normalize_illumination must be boolean")
        if not isinstance(segmentation.get("exclude_white_tissue", True), bool):
            raise TypeError("segmentation.exclude_white_tissue must be boolean")
        white_max_saturation = segmentation.get(
            "white_tissue_max_saturation", 0.25
        )
        if (
            isinstance(white_max_saturation, bool)
            or not isinstance(white_max_saturation, (int, float))
            or not 0 <= white_max_saturation <= 1
        ):
            raise ValueError(
                "segmentation.white_tissue_max_saturation must be in [0, 1]"
            )
        min_retained_fraction = segmentation.get(
            "white_tissue_min_retained_fraction", 0.50
        )
        if (
            isinstance(min_retained_fraction, bool)
            or not isinstance(min_retained_fraction, (int, float))
            or not 0 < min_retained_fraction <= 1
        ):
            raise ValueError(
                "segmentation.white_tissue_min_retained_fraction must be in (0, 1]"
            )

        features = config.get("features", {})
        if not isinstance(features, dict):
            raise TypeError("features config must be a mapping")
        valid_spaces = {"RGB", "HSV", "CIELAB", "YCbCr"}
        unknown_spaces = set(features.get("color_spaces", valid_spaces)) - valid_spaces
        if unknown_spaces:
            raise ValueError(f"Unknown color spaces: {sorted(unknown_spaces)}")

        output = config.get("output", {})
        if not isinstance(output, dict):
            raise TypeError("output config must be a mapping")
        for key in ("separate_visualization", "save_raw_table", "aggregate_cv", "write_manifest"):
            if key in output and not isinstance(output[key], bool):
                raise TypeError(f"output.{key} must be boolean")

    @staticmethod
    def _resolve_color_calibration_mode(calibration: Dict) -> str:
        if "mode" in calibration:
            return calibration["mode"]
        return "optional" if calibration.get("enabled", False) else "off"

    @staticmethod
    def _matrix_sha256(matrix: np.ndarray) -> str:
        canonical = np.asarray(matrix, dtype="<f8", order="C")
        return hashlib.sha256(canonical.tobytes()).hexdigest()

    @property
    def color_calibration_applied(self) -> bool:
        return self.color_calibration_status in {
            "applied_validated_profile", "applied_legacy_matrix",
        }

    def _resolve_calibration_path(self, configured_path: str) -> Path:
        path = Path(configured_path)
        if not path.is_absolute():
            path = Path(self.config.get("_config_dir", ".")) / path
        return path

    def _init_preprocessor(self) -> ImagePreprocessor:
        calib = self.config.get("color_calibration", {})
        imaging = self.config.get("imaging", {})
        profile_file = calib.get("profile_file")
        profile: Optional[CalibrationProfile] = None
        calibration_method = calib.get("method", "polynomial")
        polynomial_degree = calib.get("polynomial_degree", 2)
        working_domain = "encoded_srgb"

        if self.color_calibration_mode != "off" and profile_file:
            profile_path = self._resolve_calibration_path(profile_file)
            profile = load_calibration_profile(profile_path)
            self.color_calibration_profile = profile
            self.color_calibration_source = str(profile_path.resolve())
            if profile.status != "validated":
                self.color_calibration_status = "profile_not_validated"
                if self.color_calibration_mode == "required":
                    raise ValueError(
                        "required color calibration needs a validated profile"
                    )
                profile = None
            else:
                profile_preprocessing = profile.data["preprocessing"]
                mismatches = []
                if profile.data["input"]["camera_id"] != self.camera_id:
                    mismatches.append("camera_id")
                if profile_preprocessing["white_balance"] != self.default_white_balance:
                    mismatches.append("white_balance")
                if profile_preprocessing["white_balance"] == "gray_card":
                    profile_gray = np.asarray(
                        profile_preprocessing["gray_card_rgb"], dtype=np.float64
                    )
                    if (
                        self.default_gray_roi is None
                        or not np.allclose(profile_gray, self.default_gray_roi)
                    ):
                        mismatches.append("gray_card_rgb")
                if (
                    profile_preprocessing["exposure_normalization"]
                    != self.default_exposure_normalization
                ):
                    mismatches.append("exposure_normalization")
                if (
                    profile.data["input"]["kind"] == "raw"
                    and profile_preprocessing["raw_use_camera_wb"]
                    != imaging.get("raw_use_camera_wb", True)
                ):
                    mismatches.append("raw_use_camera_wb")
                if mismatches:
                    self.color_calibration_status = "profile_preprocessing_mismatch"
                    if self.color_calibration_mode == "required":
                        raise ValueError(
                            "calibration profile preprocessing does not match runtime: "
                            + ", ".join(mismatches)
                        )
                    profile = None

            if profile is not None:
                target_space = profile.data["target"]["space"]
                supported_pair = (
                    profile.input_domain == "encoded_srgb" and target_space == "sRGB"
                ) or (
                    profile.input_domain in {"linear_srgb", "camera_linear_rgb"}
                    and target_space == "XYZ"
                )
                if not supported_pair:
                    self.color_calibration_status = "profile_domain_unsupported"
                    if self.color_calibration_mode == "required":
                        raise ValueError(
                            "calibration profile input.domain and target.space are "
                            "not a supported runtime pair"
                        )
                    profile = None

            if profile is not None:
                working_domain = profile.input_domain

            if profile is not None and profile.model_type == "linear_3x3":
                calibration_method = "linear"
                polynomial_degree = 1
            elif profile is not None and profile.model_type == "legacy_polynomial":
                if profile.input_domain != "encoded_srgb":
                    raise ValueError(
                        "legacy_polynomial profiles require encoded_srgb input"
                    )
                calibration_method = "polynomial"
                polynomial_degree = profile.degree
            elif profile is not None and profile.model_type == "root_polynomial_2":
                if profile.input_domain == "encoded_srgb":
                    raise ValueError(
                        "root_polynomial_2 profiles require a linear RGB input domain"
                    )
                calibration_method = "linear"
                polynomial_degree = 1
            elif profile is not None:
                self.color_calibration_status = "profile_model_unsupported"
                if self.color_calibration_mode == "required":
                    raise ValueError(
                        f"profile model '{profile.model_type}' is not supported by this runtime"
                    )
                profile = None
                working_domain = "encoded_srgb"

        preprocessor = ImagePreprocessor(
            target_illuminant=imaging.get("target_illuminant", "D65"),
            calibration_method=calibration_method,
            polynomial_degree=polynomial_degree,
            raw_use_camera_wb=imaging.get("raw_use_camera_wb", True),
            raw_output_bps=imaging.get("bits_per_channel", 16),
            working_domain=working_domain,
        )
        if self.color_calibration_mode == "off":
            return preprocessor

        if profile is not None:
            if profile.data["target"]["space"] == "sRGB":
                preprocessor.set_color_correction_matrix(profile.matrix)
            self.color_calibration_status = "applied_validated_profile"
            self.color_calibration_matrix_sha256 = self._matrix_sha256(profile.matrix)
            return preprocessor

        if profile_file:
            return preprocessor

        legacy_requested = calib.get("enabled", False) or (
            calib.get("matrix") is not None or bool(calib.get("ccm_file"))
        )
        if not legacy_requested:
            if self.color_calibration_mode == "required":
                raise ValueError(
                    "color_calibration.mode=required requires a validated profile_file"
                )
            return preprocessor

        explicit_mode = "mode" in calib
        allow_legacy = calib.get("allow_legacy_matrix", not explicit_mode)
        if self.color_calibration_mode == "required" or not allow_legacy:
            raise ValueError(
                "legacy bare CCM matrices are not allowed; provide a validated "
                "profile_file or set allow_legacy_matrix=true in optional mode"
            )

        if calib.get("matrix") is not None:
            matrix = preprocessor.set_color_correction_matrix(calib["matrix"])
            self.color_calibration_status = "applied_legacy_matrix"
            self.color_calibration_source = "inline_matrix"
            self.color_calibration_matrix_sha256 = self._matrix_sha256(matrix)
            return preprocessor

        ccm_file = calib.get("ccm_file")
        if ccm_file:
            ccm_path = self._resolve_calibration_path(ccm_file)
            matrix = preprocessor.load_color_correction_matrix(str(ccm_path))
            self.color_calibration_status = "applied_legacy_matrix"
            self.color_calibration_source = str(ccm_path.resolve())
            self.color_calibration_matrix_sha256 = self._matrix_sha256(matrix)
            return preprocessor

        raise ValueError(
            "color_calibration.enabled=true requires color_calibration.matrix "
            "or color_calibration.ccm_file"
        )

    def _init_segmenter(self) -> BaseSegmenter:
        seg = self.config.get("segmentation", {})
        method = seg.get("method", "exg")
        kwargs = {}
        for key, value in seg.items():
            if key in {
                "method", "component_policy", "max_processing_dimension",
                "normalize_illumination", "component_min_exg",
                "exclude_white_tissue", "white_tissue_max_saturation",
                "white_tissue_min_retained_fraction",
            }:
                continue
            kwargs[_SEGMENTATION_ALIASES.get(key, key)] = value
        if method == "sam" and "model_path" in kwargs and "sam_checkpoint" not in kwargs:
            kwargs["sam_checkpoint"] = kwargs.pop("model_path")
        model_key = "sam_checkpoint" if method == "sam" else "model_path"
        if kwargs.get(model_key):
            model_path = Path(kwargs[model_key])
            if not model_path.is_absolute():
                model_path = Path(self.config.get("_config_dir", ".")) / model_path
            kwargs[model_key] = str(model_path)
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
                       white_balance: Optional[str] = None,
                       gray_roi: Optional[np.ndarray] = None,
                       return_visualization: bool = False,
                       verbose: bool = True,
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
        wb_method = white_balance or self.default_white_balance
        wb_gray_roi = gray_roi if gray_roi is not None else self.default_gray_roi
        preprocessed = self.preprocessor.process(
            str(img_path),
            white_balance_method=wb_method,
            gray_roi=wb_gray_roi,
            apply_ccm=False,
            compute_derived=False,
        )
        img_rgb = preprocessed["rgb"]
        segmentation_base_rgb = preprocessed.get("segmentation_rgb")
        if segmentation_base_rgb is None:
            segmentation_base_rgb = self.preprocessor.process_segmentation_image(
                str(img_path)
            )
        img_uint8 = (segmentation_base_rgb * 255).clip(0, 255).astype(np.uint8)

        profile = self.color_calibration_profile
        if profile is not None and self.color_calibration_applied:
            actual_kind = (
                "raw"
                if img_path.suffix.lower() in RAW_IMAGE_EXTENSIONS
                else "rendered_rgb"
            )
            profile_kind = profile.data["input"]["kind"]
            if profile_kind == "jpeg":
                profile_kind = "rendered_rgb"
            if actual_kind != profile_kind:
                raise ValueError(
                    f"calibration profile expects {profile_kind} input, "
                    f"but {img_path.name} is {actual_kind}"
                )
            if wb_method != profile.data["preprocessing"]["white_balance"]:
                raise ValueError(
                    "white-balance override does not match active calibration profile"
                )
            if wb_method == "gray_card":
                profile_gray = np.asarray(
                    profile.data["preprocessing"]["gray_card_rgb"], dtype=np.float64
                )
                if wb_gray_roi is None or not np.allclose(profile_gray, wb_gray_roi):
                    raise ValueError(
                        "gray-card override does not match active calibration profile"
                    )

        # ---- Step 2: 分割 ----
        segmentation_rgb = segmentation_base_rgb
        if self.normalize_segmentation_illumination:
            segmentation_rgb = self.preprocessor.white_balance_gray_world(
                segmentation_base_rgb
            )
        raw_mask = self._segment_image(segmentation_rgb)
        component_mask = self._select_analysis_mask(
            raw_mask,
            self.component_policy,
            img_rgb=segmentation_rgb,
            min_exg=self.component_min_exg,
        )
        if self.exclude_white_tissue:
            mask, white_tissue_qc = self._exclude_white_tissue(
                component_mask,
                segmentation_base_rgb,
                max_saturation=self.white_tissue_max_saturation,
                min_retained_fraction=self.white_tissue_min_retained_fraction,
            )
        else:
            mask = component_mask
            white_tissue_qc = {
                "QC_white_tissue_removed_px": 0.0,
                "QC_white_tissue_removed_fraction": 0.0,
                "QC_white_tissue_filter_rollback": 0.0,
            }
        mask_qc = self._mask_qc(
            mask,
            raw_mask=raw_mask,
            selected_component_mask=component_mask,
        )
        mask_qc.update(white_tissue_qc)

        # Feature extraction works on the leaf bounding box at original pixel
        # resolution. This preserves color values while avoiding repeated
        # full-frame transforms when the leaf occupies only a small area.
        feature_views = self._crop_to_mask(
            mask,
            rgb=img_rgb,
        )
        feature_mask = feature_views.pop("mask")
        calibration_qc = {
            "QC_CCM_applied": 0.0,
            "QC_CCM_negative_fraction": 0.0,
            "QC_CCM_clipped_fraction": 0.0,
            "QC_CCM_R_clipped_fraction": 0.0,
            "QC_CCM_G_clipped_fraction": 0.0,
            "QC_CCM_B_clipped_fraction": 0.0,
        }
        if (
            profile is not None
            and self.color_calibration_applied
            and profile.data["target"]["space"] == "XYZ"
        ):
            application = apply_calibration_profile(
                profile, feature_views["rgb"], mask=feature_mask
            )
            feature_views["rgb"] = application.srgb
            feature_views["rgb_uint8"] = (
                application.srgb * 255
            ).clip(0, 255).astype(np.uint8)
            feature_lab = application.lab_d65
            calibration_qc.update(application.qc)
            calibration_qc["QC_CCM_applied"] = 1.0
        elif self.preprocessor.has_color_correction_matrix:
            feature_views["rgb"] = self.preprocessor.apply_color_correction(
                feature_views["rgb"]
            )
            feature_views["rgb_uint8"] = (
                feature_views["rgb"] * 255
            ).clip(0, 255).astype(np.uint8)
            feature_lab = rgb_to_lab(feature_views["rgb"])
            calibration_qc["QC_CCM_applied"] = 1.0
        else:
            feature_views["rgb_uint8"] = (
                feature_views["rgb"] * 255
            ).clip(0, 255).astype(np.uint8)
            feature_lab = rgb_to_lab(feature_views["rgb"])
        feature_hsv = rgb_to_hsv(feature_views["rgb"])

        # ---- Step 3: 颜色特征 ----
        color_feats = self.color_extractor.extract(
            feature_views["rgb"],
            feature_mask,
            precomputed={
                "CIELAB": feature_lab,
                "HSV": feature_hsv,
            },
        )

        # ---- Step 4: 植被指数 ----
        veg_feats = self.veg_index_extractor.compute(feature_views["rgb"], feature_mask)

        # ---- Step 5: 纹理特征 ----
        texture_feats = self.texture_extractor.compute(
            feature_views["rgb_uint8"], feature_mask
        )

        # ---- Step 6: 形状特征 ----
        # Connected-component selection has already happened. If white tissue
        # splits the remaining green tissue into several pieces, keep all of
        # those pieces consistent with the mask used by the other extractors.
        shape_feats = self.shape_extractor.compute(
            mask, component_policy="all"
        )

        # ---- Step 7: 颜色均匀性 ----
        uniformity_feats = self.color_texture_analyzer.color_uniformity(
            feature_lab, feature_mask
        )

        # ---- 汇总所有特征 ----
        all_features: Dict[str, float] = {}
        all_features.update(color_feats)
        all_features.update(veg_feats)
        all_features.update(texture_feats)
        all_features.update(shape_feats)
        all_features.update(uniformity_feats)
        all_features.update(mask_qc)
        all_features.update(calibration_qc)

        # ---- 元数据 ----
        sid = sample_id or parse_sample_id(img_path.name)
        if metadata is not None:
            if not isinstance(metadata, dict):
                raise TypeError("metadata must be a dictionary")
            reserved_metadata_keys = {
                "sample_id", "image_path", "replicate", "developmental_stage",
                "features", "metadata", "mask", "visualization",
            }
            collisions = sorted(
                set(metadata) & (reserved_metadata_keys | set(all_features))
            )
            if collisions:
                raise ValueError(
                    "metadata keys collide with reserved/result fields: "
                    f"{collisions}"
                )
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
                img_uint8,
                mask,
                all_features,
                excluded_white_mask=(component_mask > 0) & (mask == 0),
            )

        elapsed = time.time() - t_start
        if verbose:
            print(
                f"  [{sid}] Processed in {elapsed:.2f}s → "
                f"{len(all_features)} features extracted"
            )

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
                      white_balance: Optional[str] = None,
                      gray_roi: Optional[np.ndarray] = None,
                      save_visualizations: Optional[bool] = None,
                      visualization_dir: Optional[str] = None,
                      verbose: bool = True,
                      progress_callback: Optional[Callable[[Dict[str, object]], None]] = None,
                      cancel_check: Optional[Callable[[], bool]] = None,
                      ) -> pd.DataFrame:
        """批量处理目录下的所有图像.

        Args:
            image_dir: 图像目录
            output_dir: 结果输出目录 (可选)
            output_csv: 输出CSV文件名 (含路径)
            id_pattern: 样本ID提取正则模式, None=自动
            group_by_sample: 是否按样本ID汇总多张图像
            white_balance: 白平衡方法
            verbose: 是否打印进度
            progress_callback: 每张图开始、成功或失败时接收进度字典
            cancel_check: 返回 True 时在下一张图开始前安全停止

        Returns:
            DataFrame: 表型表
        """
        img_paths = find_images(image_dir)
        if save_visualizations is None:
            save_visualizations = self.default_save_visualizations
        self.last_batch_failures = []
        self.last_batch_cancelled = False
        if verbose:
            print(f"Found {len(img_paths)} images in {image_dir}")

        def emit_progress(status: str, index: int, path: Optional[Path], message: str) -> None:
            if progress_callback is None:
                return
            event: Dict[str, object] = {
                "status": status,
                "current": index,
                "total": len(img_paths),
                "image_path": str(path) if path is not None else "",
                "message": message,
                "successful": len(records),
                "failed": len(self.last_batch_failures),
            }
            try:
                progress_callback(event)
            except Exception as exc:
                if verbose:
                    print(f"  WARNING: progress callback failed: {exc}")

        if not img_paths:
            print("WARNING: No images found!")
            return pd.DataFrame()

        vis_dir = None
        image_root = Path(image_dir).resolve()
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
            if cancel_check is not None and cancel_check():
                self.last_batch_cancelled = True
                emit_progress("cancelled", i, img_path, "用户已取消，正在保存已完成结果")
                break
            if verbose:
                print(f"[{i+1}/{len(img_paths)}] Processing {img_path.name}...")
            emit_progress("processing", i + 1, img_path, f"正在处理 {img_path.name}")

            try:
                sid = parse_sample_id(img_path.name, id_pattern) if id_pattern else None
                result = self.process_single(
                    str(img_path),
                    sample_id=sid,
                    white_balance=white_balance,
                    gray_roi=gray_roi,
                    return_visualization=save_visualizations,
                    verbose=verbose,
                )
                if save_visualizations and vis_dir is not None and "visualization" in result:
                    vis_path = self._visualization_output_path(
                        vis_dir, image_root, img_path
                    )
                    write_image_rgb(vis_path, result["visualization"])
                    result.pop("visualization", None)
                    result.pop("mask", None)
                records.append(result)
                emit_progress("success", i + 1, img_path, f"已完成 {img_path.name}")
            except Exception as e:
                print(f"  ERROR processing {img_path.name}: {e}")
                self.last_batch_failures.append({
                    "image_path": str(img_path),
                    "error_type": type(e).__name__,
                    "error": str(e),
                })
                emit_progress("failed", i + 1, img_path, str(e))
                continue

        if not records:
            if (self.last_batch_failures or self.last_batch_cancelled) and (output_csv or output_dir):
                base_path = Path(output_csv or str(
                    Path(output_dir or ".") / self.output_table_name
                ))
                if not base_path.suffix:
                    base_path = base_path.with_suffix(".csv")
                if self.last_batch_failures:
                    failure_path = base_path.with_name(f"{base_path.stem}_failures.csv")
                    safe_mkdir(failure_path.parent)
                    pd.DataFrame(self.last_batch_failures).to_csv(failure_path, index=False)
                    print(f"Failure report saved to: {failure_path}")
                if self.write_manifest:
                    self._save_run_manifest(
                        output_path=base_path,
                        image_dir=image_dir,
                        discovered_images=len(img_paths),
                        successful_images=0,
                        output_samples=0,
                    )
            return pd.DataFrame()

        # 构建DataFrame
        rows = []
        trait_columns = set()
        for rec in records:
            row = {"sample_id": rec["sample_id"],
                   "image_path": rec["image_path"]}
            for key in ("replicate", "developmental_stage"):
                if key in rec:
                    row[key] = rec[key]
            row.update(rec.get("metadata") or {})
            row.update(rec["features"])
            trait_columns.update(rec["features"].keys())
            rows.append(row)

        raw_df = pd.DataFrame(rows)
        df = raw_df

        # 按样本ID汇总 (多重复/)
        if group_by_sample:
            df = self._aggregate_by_sample(
                raw_df,
                trait_columns=sorted(trait_columns),
                include_cv=self.aggregate_cv,
            )

        # 保存
        if output_csv or output_dir:
            output_path = output_csv or str(
                Path(output_dir or ".") / self.output_table_name
            )
            output_path = self._save_table(df, output_path)
            if group_by_sample and self.save_raw_table:
                raw_path = output_path.with_name(
                    f"{output_path.stem}_raw{output_path.suffix}"
                )
                self._save_table(raw_df, str(raw_path))
                if verbose:
                    print(f"  Per-image table saved to: {raw_path}")
            if self.last_batch_failures:
                failure_path = output_path.with_name(f"{output_path.stem}_failures.csv")
                pd.DataFrame(self.last_batch_failures).to_csv(failure_path, index=False)
                print(f"  Failure report saved to: {failure_path}")
            if self.write_manifest:
                manifest_path = self._save_run_manifest(
                    output_path=output_path,
                    image_dir=image_dir,
                    discovered_images=len(img_paths),
                    successful_images=len(records),
                    output_samples=len(df),
                )
                if verbose:
                    print(f"  Run manifest saved to: {manifest_path}")
            print(f"\nPhenotype table saved to: {output_path}")
            print(f"  Shape: {df.shape[0]} samples × {df.shape[1]} traits")

        return df

    # ----------------------------------------------------------
    # 按样本汇总
    # ----------------------------------------------------------
    def _segment_image(self, img_rgb: np.ndarray) -> np.ndarray:
        """Segment on a bounded-size proxy and return a full-resolution mask."""
        height, width = img_rgb.shape[:2]
        max_dimension = self.max_segmentation_dimension
        if max_dimension is None or max(height, width) <= max_dimension:
            return self.segmenter.segment(img_rgb)

        scale = max_dimension / max(height, width)
        resized = cv2.resize(
            img_rgb,
            (max(1, round(width * scale)), max(1, round(height * scale))),
            interpolation=cv2.INTER_AREA,
        )
        small_mask = self.segmenter.segment(resized)
        mask = cv2.resize(small_mask, (width, height), interpolation=cv2.INTER_NEAREST)
        return (mask > 0).astype(np.uint8) * 255

    @staticmethod
    def _select_analysis_mask(
        mask: np.ndarray,
        policy: str,
        img_rgb: Optional[np.ndarray] = None,
        min_exg: float = 0.30,
    ) -> np.ndarray:
        """Apply the configured connected-component policy consistently."""
        if policy not in {"largest", "all"}:
            raise ValueError(f"Unknown component policy: {policy}")
        mask_bin = (mask > (127 if mask.max() > 1 else 0)).astype(np.uint8)
        if not np.any(mask_bin):
            return mask_bin * 255
        if policy == "all":
            return mask_bin * 255

        count, labels, stats, _ = cv2.connectedComponentsWithStats(mask_bin, 8)
        candidate_labels = list(range(1, count))
        if img_rgb is not None:
            if img_rgb.shape[:2] != mask_bin.shape:
                raise ValueError("mask and img_rgb must have matching height and width")
            red, green, blue = (
                img_rgb[..., 0], img_rgb[..., 1], img_rgb[..., 2]
            )
            total = red + green + blue + 1e-10
            exg = (2 * green - red - blue) / total
            vegetation_labels = [
                label for label in candidate_labels
                if float(exg[labels == label].mean()) >= min_exg
            ]
            if not vegetation_labels:
                return np.zeros_like(mask_bin, dtype=np.uint8)
            candidate_labels = vegetation_labels

        if not candidate_labels:
            return np.zeros_like(mask_bin, dtype=np.uint8)

        selected_label = max(
            candidate_labels, key=lambda label: stats[label, cv2.CC_STAT_AREA]
        )
        return (labels == selected_label).astype(np.uint8) * 255

    @staticmethod
    def _exclude_white_tissue(
        mask: np.ndarray,
        img_rgb: np.ndarray,
        max_saturation: float = 0.25,
        min_retained_fraction: float = 0.50,
    ) -> tuple[np.ndarray, Dict[str, float]]:
        """Remove low-saturation white/gray tissue from inside a leaf mask.

        Saturation is exposure-tolerant, so the same threshold can identify a
        pale midrib in both bright and dark photographs. Processing is limited
        to the leaf bounding box to avoid allocating full-frame color helpers.
        If filtering would remove too much of the selected leaf, the original
        mask is returned and the rollback is exposed through QC.
        """
        if mask.shape != img_rgb.shape[:2]:
            raise ValueError("mask and img_rgb must have matching height and width")
        if not 0 <= max_saturation <= 1:
            raise ValueError("max_saturation must be in [0, 1]")
        if not 0 < min_retained_fraction <= 1:
            raise ValueError("min_retained_fraction must be in (0, 1]")

        mask_bin = mask > (127 if mask.max() > 1 else 0)
        original_area = int(mask_bin.sum())
        empty_qc = {
            "QC_white_tissue_removed_px": 0.0,
            "QC_white_tissue_removed_fraction": 0.0,
            "QC_white_tissue_filter_rollback": 0.0,
        }
        if original_area == 0:
            return mask_bin.astype(np.uint8) * 255, empty_qc

        x, y, width, height = cv2.boundingRect(mask_bin.astype(np.uint8))
        mask_crop = mask_bin[y:y + height, x:x + width]
        rgb_crop = np.clip(
            img_rgb[y:y + height, x:x + width].astype(np.float32), 0.0, 1.0
        )
        channel_max = rgb_crop.max(axis=2)
        channel_min = rgb_crop.min(axis=2)
        saturation = np.divide(
            channel_max - channel_min,
            channel_max,
            out=np.zeros_like(channel_max),
            where=channel_max > 1e-8,
        )
        excluded_crop = mask_crop & (saturation <= max_saturation)
        removed_area = int(excluded_crop.sum())
        if removed_area == 0:
            return mask_bin.astype(np.uint8) * 255, empty_qc

        retained_fraction = (original_area - removed_area) / original_area
        if retained_fraction < min_retained_fraction:
            rollback_qc = dict(empty_qc)
            rollback_qc["QC_white_tissue_filter_rollback"] = 1.0
            return mask_bin.astype(np.uint8) * 255, rollback_qc

        refined = mask_bin.copy()
        refined_crop = refined[y:y + height, x:x + width]
        refined_crop[excluded_crop] = False
        qc = {
            "QC_white_tissue_removed_px": float(removed_area),
            "QC_white_tissue_removed_fraction": float(removed_area / original_area),
            "QC_white_tissue_filter_rollback": 0.0,
        }
        return refined.astype(np.uint8) * 255, qc

    @staticmethod
    def _crop_to_mask(mask: np.ndarray, **arrays: np.ndarray) -> Dict[str, np.ndarray]:
        """Crop aligned image arrays to the selected mask bounding box."""
        mask_bin = mask > (127 if mask.max() > 1 else 0)
        if not np.any(mask_bin):
            cropped = {name: array[:1, :1] for name, array in arrays.items()}
            cropped["mask"] = mask[:1, :1]
            return cropped
        x, y, width, height = cv2.boundingRect(mask_bin.astype(np.uint8))
        cropped = {
            name: array[y:y + height, x:x + width]
            for name, array in arrays.items()
        }
        cropped["mask"] = mask[y:y + height, x:x + width]
        return cropped

    @staticmethod
    def _mask_qc(
        mask: np.ndarray,
        raw_mask: Optional[np.ndarray] = None,
        selected_component_mask: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """Summarize segmentation mask quality for downstream filtering."""
        if mask.max() > 1:
            mask_bin = mask > 127
        else:
            mask_bin = mask > 0

        area_px = int(mask_bin.sum())
        total_px = int(mask_bin.size)
        raw_mask = mask if raw_mask is None else raw_mask
        raw_bin = raw_mask > (127 if raw_mask.max() > 1 else 0)
        component_count, _, component_stats, _ = cv2.connectedComponentsWithStats(
            raw_bin.astype(np.uint8), 8
        )
        component_areas = component_stats[1:, cv2.CC_STAT_AREA]
        raw_area_px = int(raw_bin.sum())
        largest_component_area = int(component_areas.max()) if component_areas.size else 0
        selected_component_mask = (
            mask if selected_component_mask is None else selected_component_mask
        )
        selected_component_bin = selected_component_mask > (
            127 if selected_component_mask.max() > 1 else 0
        )
        selected_component_area_px = int(selected_component_bin.sum())

        border_pixels = np.zeros_like(mask_bin, dtype=bool)
        border_pixels[0, :] = True
        border_pixels[-1, :] = True
        border_pixels[:, 0] = True
        border_pixels[:, -1] = True
        border_contact_px = int(np.count_nonzero(mask_bin & border_pixels))
        feats = {
            "QC_mask_area_px": float(area_px),
            "QC_mask_area_ratio": float(area_px / total_px) if total_px else np.nan,
            "QC_mask_is_empty": float(area_px == 0),
            "QC_raw_mask_area_px": float(raw_area_px),
            "QC_raw_mask_area_ratio": (
                float(raw_area_px / total_px) if total_px else np.nan
            ),
            "QC_component_count": float(max(0, component_count - 1)),
            "QC_largest_component_area_px": float(largest_component_area),
            "QC_largest_component_fraction": (
                float(largest_component_area / raw_area_px) if raw_area_px else np.nan
            ),
            "QC_selected_component_fraction": (
                float(selected_component_area_px / raw_area_px)
                if raw_area_px else np.nan
            ),
            "QC_border_contact_ratio": (
                float(border_contact_px / area_px) if area_px else np.nan
            ),
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
    def _visualization_output_path(
        visualization_dir: Path,
        image_root: Path,
        image_path: Path,
    ) -> Path:
        """Return a collision-resistant visualization path for recursive input."""
        resolved_image = image_path.resolve()
        try:
            relative_image = resolved_image.relative_to(image_root.resolve())
        except ValueError:
            relative_image = Path(resolved_image.name)
        extension_token = relative_image.suffix.lstrip(".") or "image"
        filename = f"{relative_image.stem}__{extension_token}_vis.png"
        return Path(visualization_dir) / relative_image.parent / filename

    @staticmethod
    def _metadata_value_key(value: object) -> str:
        """Create a stable comparison key for scalar or JSON-like metadata."""
        try:
            missing = pd.isna(value)
            if isinstance(missing, (bool, np.bool_)) and missing:
                return "__missing__"
        except (TypeError, ValueError):
            pass
        return json.dumps(
            value,
            sort_keys=True,
            ensure_ascii=False,
            default=str,
            separators=(",", ":"),
        )

    @classmethod
    def _validate_metadata_consistency(
        cls,
        df: pd.DataFrame,
        metadata_columns: List[str],
    ) -> None:
        """Reject sample groups whose experimental metadata is inconsistent."""
        conflicts = []
        for column in metadata_columns:
            for sample_id, values in df.groupby(
                "sample_id", dropna=False, sort=True
            )[column]:
                unique_values = {
                    cls._metadata_value_key(value) for value in values.tolist()
                }
                if len(unique_values) > 1:
                    conflicts.append(f"{sample_id!r}:{column}")
        if conflicts:
            preview = ", ".join(conflicts[:10])
            suffix = " ..." if len(conflicts) > 10 else ""
            raise ValueError(
                "Conflicting metadata within sample groups: "
                f"{preview}{suffix}. Use distinct sample IDs or --no-aggregate."
            )

    @staticmethod
    def _aggregate_by_sample(df: pd.DataFrame,
                             trait_columns: Optional[List[str]] = None,
                             include_cv: bool = False,
                             ) -> pd.DataFrame:
        """按样本ID汇总多重复测量.

        对数值型特征计算: mean, std, cv (变异系数)
        对非数值实验元数据: 要求组内一致后取第一个值
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

        groupby = df.groupby("sample_id", dropna=False, sort=True)
        group_sizes = groupby.size()

        # Do not manufacture hundreds of all-NaN replicate-stat columns when
        # every sample occurs only once.
        if group_sizes.max() <= 1:
            result = df.copy()
            result["n_replicates"] = 1
            if "image_path" in result.columns:
                result["image_paths"] = result["image_path"].astype(str)
            return result

        missing_traits = [col for col in trait_columns if col not in df.columns]
        if missing_traits:
            raise ValueError(f"Unknown trait columns: {missing_traits}")

        metadata_columns = [
            col for col in df.columns
            if col not in {"sample_id", "image_path", "replicate"}
            and col not in trait_columns
        ]
        if metadata_columns:
            LeafColorPipeline._validate_metadata_consistency(
                df, metadata_columns
            )

        if not trait_columns:
            grouped = (
                groupby[metadata_columns].first()
                if metadata_columns
                else pd.DataFrame(index=group_sizes.index)
            )
            grouped["n_replicates"] = group_sizes
            if "image_path" in df.columns:
                grouped = grouped.join(
                    groupby["image_path"].first().rename("image_path")
                )
                grouped = grouped.join(
                    groupby["image_path"].agg(
                        lambda values: ";".join(str(value) for value in values)
                    ).rename("image_paths")
                )
            return grouped.reset_index()

        agg_dict = {col: ["mean", "std"] for col in trait_columns}
        grouped = groupby.agg(agg_dict)

        # 扁平化列名. Keep per-image feature names intact; add rep_* for replicate stats
        grouped.columns = [
            col[0] if col[1] == "mean" else f"{col[0]}_rep_{col[1]}"
            for col in grouped.columns
        ]

        # CV is opt-in because it is undefined/unstable for signed, zero-centred,
        # categorical, histogram, and QC traits.
        if include_cv:
            cv_columns = {}
            for col in trait_columns:
                if col.startswith(("QC_", "Hist_")):
                    continue
                mean_col = col
                std_col = f"{col}_rep_std"
                if mean_col in grouped.columns and std_col in grouped.columns:
                    denominator = grouped[mean_col].abs()
                    cv_columns[f"{col}_rep_cv"] = (
                        grouped[std_col] / denominator.where(denominator > 1e-6)
                    )
            if cv_columns:
                grouped = pd.concat(
                    [grouped, pd.DataFrame(cv_columns, index=grouped.index)], axis=1
                )

        # 添加重复数
        n_replicates = group_sizes.rename("n_replicates")
        grouped = pd.concat([grouped, n_replicates], axis=1).copy()

        # 保留图像路径、发育阶段和用户元数据等非性状列。
        if metadata_columns:
            grouped = grouped.join(groupby[metadata_columns].first())
        if "image_path" in df.columns:
            grouped = grouped.join(
                groupby["image_path"].first().rename("image_path")
            )
            image_paths = groupby["image_path"].agg(
                lambda values: ";".join(str(value) for value in values)
            ).rename("image_paths")
            grouped = grouped.join(image_paths)

        return grouped.reset_index()

    def _save_table(self, df: pd.DataFrame, output_path: str) -> Path:
        """Save a phenotype table using the configured or explicit file format."""
        path = Path(output_path)
        suffix_to_format = {
            ".csv": "csv",
            ".xlsx": "excel",
            ".json": "json",
        }
        output_format = suffix_to_format.get(path.suffix.lower(), self.output_format)
        if path.suffix and path.suffix.lower() not in suffix_to_format:
            raise ValueError(
                f"Unsupported output extension '{path.suffix}'; use .csv, .json, or .xlsx"
            )
        if not path.suffix:
            extension = {"csv": ".csv", "excel": ".xlsx", "json": ".json"}[output_format]
            path = path.with_suffix(extension)

        safe_mkdir(path.parent)
        if output_format == "csv":
            df.to_csv(path, index=False)
        elif output_format == "json":
            df.to_json(path, orient="records", indent=2, force_ascii=False)
        else:
            try:
                df.to_excel(path, index=False)
            except ImportError as exc:
                raise ImportError("Excel output requires openpyxl>=3.1") from exc
        return path

    def _save_run_manifest(
        self,
        output_path: Path,
        image_dir: str,
        discovered_images: int,
        successful_images: int,
        output_samples: int,
    ) -> Path:
        """Write a reproducibility sidecar for a completed batch run."""
        serialized_config = json.dumps(
            self.config, sort_keys=True, ensure_ascii=False, default=str
        )
        dependency_versions = {}
        for distribution in (
            "numpy", "opencv-contrib-python-headless", "opencv-python-headless",
            "opencv-python", "pandas",
            "scikit-image", "rawpy", "torch", "segmentation-models-pytorch",
        ):
            try:
                dependency_versions[distribution] = version(distribution)
            except PackageNotFoundError:
                continue

        manifest = {
            "schema_version": 1,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "status": (
                "cancelled" if self.last_batch_cancelled
                else "completed_with_errors" if self.last_batch_failures
                else "completed"
            ),
            "cancelled": self.last_batch_cancelled,
            "input_dir": str(Path(image_dir).resolve()),
            "output_table": str(output_path.resolve()),
            "discovered_images": int(discovered_images),
            "successful_images": int(successful_images),
            "failed_images": int(len(self.last_batch_failures)),
            "output_samples": int(output_samples),
            "component_policy": self.component_policy,
            "color_calibration": self._color_calibration_manifest(),
            "config_sha256": hashlib.sha256(serialized_config.encode("utf-8")).hexdigest(),
            "config": self.config,
            "runtime": {
                "python": sys.version,
                "platform": platform.platform(),
                "dependencies": dependency_versions,
            },
        }
        manifest_path = output_path.with_name(f"{output_path.stem}_manifest.json")
        safe_mkdir(manifest_path.parent)
        manifest_path.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )
        return manifest_path

    def _color_calibration_manifest(self) -> Dict[str, object]:
        profile = self.color_calibration_profile
        if profile is not None and self.color_calibration_applied:
            matrix = profile.matrix
        else:
            matrix = self.preprocessor._ccm
        return {
            "mode": self.color_calibration_mode,
            "status": self.color_calibration_status,
            "applied": self.color_calibration_applied,
            "source": self.color_calibration_source,
            "matrix_sha256": self.color_calibration_matrix_sha256,
            "matrix": matrix.tolist() if matrix is not None else None,
            "profile_id": profile.profile_id if profile is not None else None,
            "profile_sha256": profile.sha256 if profile is not None else None,
            "input_domain": profile.input_domain if profile is not None else None,
            "input": deepcopy(profile.data["input"]) if profile is not None else None,
            "preprocessing": (
                deepcopy(profile.data["preprocessing"])
                if profile is not None else None
            ),
            "model": deepcopy(profile.data["model"]) if profile is not None else None,
            "target": deepcopy(profile.data["target"]) if profile is not None else None,
            "reference": deepcopy(profile.data["reference"]) if profile is not None else None,
            "datasets": deepcopy(profile.data.get("datasets")) if profile is not None else None,
            "quality": deepcopy(profile.data["quality"]) if profile is not None else None,
        }

    # ----------------------------------------------------------
    # 可视化
    # ----------------------------------------------------------
    @staticmethod
    def _create_visualization(img_uint8: np.ndarray,
                              mask: np.ndarray,
                              features: Dict[str, float],
                              excluded_white_mask: Optional[np.ndarray] = None,
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

        # Mark excluded low-saturation petiole/midrib pixels in red so users
        # can verify the tissue filter without inspecting the numeric mask.
        if excluded_white_mask is not None and np.any(excluded_white_mask):
            excluded = excluded_white_mask.astype(bool)
            red_overlay = vis.copy()
            red_overlay[excluded] = (255, 0, 0)
            vis = cv2.addWeighted(vis, 0.35, red_overlay, 0.65, 0)
            excluded_contours, _ = cv2.findContours(
                excluded.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(
                vis, excluded_contours, -1, (255, 0, 0), max(1, w // 600)
            )

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
            (
                "White rm%",
                100 * features.get("QC_white_tissue_removed_fraction", 0),
                "{:.1f}",
            ),
        ]
        for label, val, fmt in key_items:
            if np.isnan(val):
                continue
            text = f"{label}: {fmt.format(val)}"
            cv2.putText(vis, text, (10, y), font, scale, (0, 0, 0), outline_thickness)
            cv2.putText(vis, text, (10, y), font, scale, (255, 255, 255), text_thickness)
            y += line_step

        return vis
