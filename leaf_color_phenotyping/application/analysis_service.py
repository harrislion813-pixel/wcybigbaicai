from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from threading import Event
from typing import Any, Dict

import yaml

from src.color_calibration import load_calibration_profile
from src.pipeline import LeafColorPipeline
from src.utils import find_images

from .models import (
    AnalysisRequest, AnalysisResult, PreviewItem, ProgressCallback, ProgressEvent,
)


class AnalysisService:
    def __init__(self, project_dir: str | Path | None = None):
        self.project_dir = Path(project_dir or Path(__file__).resolve().parent.parent)
        self.config_path = self.project_dir / "config.yaml"

    def build_config(self, request: AnalysisRequest) -> Dict[str, Any]:
        if not request.input_dir.is_dir():
            raise FileNotFoundError(f"图片文件夹不存在: {request.input_dir}")
        request.output_dir.mkdir(parents=True, exist_ok=True)
        with self.config_path.open("r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle) or {}
        config = deepcopy(config)
        config["_config_dir"] = str(self.project_dir.resolve())
        config.setdefault("input", {})["image_dir"] = str(request.input_dir.resolve())
        config["input"]["output_dir"] = str(request.output_dir.resolve())
        config.setdefault("segmentation", {})["method"] = request.segmentation_method
        config["segmentation"]["device"] = request.device
        config["segmentation"]["exclude_white_tissue"] = request.exclude_white_tissue
        output = config.setdefault("output", {})
        output["format"] = request.output_format
        output["separate_visualization"] = request.save_visualizations
        output["write_manifest"] = True

        calibration = config.setdefault("color_calibration", {})
        if request.calibration_mode == "relative":
            calibration.update({
                "mode": "off", "profile_file": "", "allow_legacy_matrix": False,
            })
        else:
            if request.profile_path is None:
                raise ValueError("跨批次分析必须选择经过验证的颜色 Profile")
            profile = load_calibration_profile(request.profile_path)
            if profile.status != "validated":
                raise ValueError("跨批次分析不能使用 draft 或无效 Profile")
            calibration.update({
                "mode": "required",
                "profile_file": str(Path(request.profile_path).resolve()),
                "allow_legacy_matrix": False,
            })
            image = config.setdefault("imaging", {})
            preprocessing = profile.data["preprocessing"]
            image.update({
                "camera_id": profile.data["input"]["camera_id"],
                "white_balance": preprocessing["white_balance"],
                "exposure_normalization": preprocessing["exposure_normalization"],
                "raw_use_camera_wb": preprocessing["raw_use_camera_wb"],
            })
            if preprocessing.get("gray_card_rgb") is not None:
                image["gray_card_rgb"] = preprocessing["gray_card_rgb"]
            else:
                image.pop("gray_card_rgb", None)
        return config

    @staticmethod
    def choose_preview_images(paths: list[Path], count: int = 5) -> list[Path]:
        if len(paths) <= count:
            return paths
        indices = sorted({round(index * (len(paths) - 1) / (count - 1)) for index in range(count)})
        return [paths[index] for index in indices]

    def preview(self, request: AnalysisRequest, count: int = 5) -> list[PreviewItem]:
        paths = find_images(str(request.input_dir))
        if not paths:
            raise ValueError("所选文件夹中没有支持的图片")
        pipeline = LeafColorPipeline(self.build_config(request))
        items = []
        for path in self.choose_preview_images(paths, count=count):
            result = pipeline.process_single(
                str(path), return_visualization=True, verbose=False
            )
            features = result["features"]
            warnings = []
            if features.get("QC_mask_is_empty", 0) >= 1:
                warnings.append("未识别到叶片")
            area = float(features.get("QC_mask_area_ratio", 0))
            if area < 0.002:
                warnings.append("叶片面积比例过小")
            if area > 0.85:
                warnings.append("叶片面积比例异常偏大")
            items.append(PreviewItem(
                image_path=path,
                visualization=result["visualization"],
                features=features,
                warnings=tuple(warnings),
            ))
        return items

    def run(
        self,
        request: AnalysisRequest,
        *,
        progress_callback: ProgressCallback | None = None,
        cancel_event: Event | None = None,
    ) -> AnalysisResult:
        config = self.build_config(request)
        pipeline = LeafColorPipeline(config)
        extension = {"csv": ".csv", "excel": ".xlsx", "json": ".json"}[
            request.output_format
        ]
        table_path = request.output_dir / f"leaf_color_phenotypes{extension}"

        def on_progress(event: Dict[str, object]) -> None:
            if progress_callback is None:
                return
            raw_path = str(event.get("image_path") or "")
            progress_callback(ProgressEvent(
                current=int(event.get("current", 0)),
                total=int(event.get("total", 0)),
                image_path=Path(raw_path) if raw_path else None,
                status=str(event.get("status", "")),
                message=str(event.get("message", "")),
                successful=int(event.get("successful", 0)),
                failed=int(event.get("failed", 0)),
            ))

        dataframe = pipeline.process_batch(
            str(request.input_dir),
            output_csv=str(table_path),
            id_pattern=request.id_pattern,
            group_by_sample=request.group_by_sample,
            save_visualizations=request.save_visualizations,
            verbose=False,
            progress_callback=on_progress,
            cancel_check=cancel_event.is_set if cancel_event is not None else None,
        )
        raw_path = table_path.with_name(f"{table_path.stem}_raw{table_path.suffix}")
        manifest_path = table_path.with_name(f"{table_path.stem}_manifest.json")
        failure_path = table_path.with_name(f"{table_path.stem}_failures.csv")
        return AnalysisResult(
            dataframe=dataframe,
            table_path=table_path,
            raw_table_path=raw_path if raw_path.is_file() else None,
            manifest_path=manifest_path if manifest_path.is_file() else None,
            failure_path=failure_path if failure_path.is_file() else None,
            visualization_dir=(request.output_dir / "visualizations")
            if request.save_visualizations else None,
            failures=tuple(pipeline.last_batch_failures),
            cancelled=pipeline.last_batch_cancelled,
        )

