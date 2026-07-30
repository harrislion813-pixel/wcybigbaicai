from __future__ import annotations

import csv
from pathlib import Path

from src.calibration_workflow import (
    CalibrationProfileRequest, fit_calibration_profile,
)
from src.colorchecker_detection import (
    extract_colorchecker_patches, load_calibration_image,
)
from src.utils import COLORCHECKER_24_PATCH_IDS

from .models import CalibrationImageRequest, CalibrationImageResult


class CalibrationService:
    @staticmethod
    def _write_patch_csv(path: Path, rgb) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["patch_id", "R", "G", "B"])
            for patch_id, values in zip(COLORCHECKER_24_PATCH_IDS, rgb):
                writer.writerow([patch_id, *[f"{float(value):.10g}" for value in values]])
        return path

    @staticmethod
    def load_display_image(path: str | Path, input_domain: str = "linear_srgb"):
        _, display, _ = load_calibration_image(path, input_domain=input_domain)
        return display

    def create_profile(self, request: CalibrationImageRequest) -> CalibrationImageResult:
        training_working, training_display, training_kind = load_calibration_image(
            request.training_image,
            input_domain=request.input_domain,
            raw_use_camera_wb=request.raw_use_camera_wb,
        )
        validation_working, validation_display, validation_kind = load_calibration_image(
            request.validation_image,
            input_domain=request.input_domain,
            raw_use_camera_wb=request.raw_use_camera_wb,
        )
        if training_kind != validation_kind:
            raise ValueError("训练色卡图和验证色卡图必须都是 RAW，或都是普通图片")
        training = extract_colorchecker_patches(
            training_working, display_rgb=training_display,
            corners=request.training_corners,
        )
        validation = extract_colorchecker_patches(
            validation_working, display_rgb=validation_display,
            corners=request.validation_corners,
        )
        output = Path(request.output_path)
        stem = output.name.removesuffix(".ccm.json")
        training_csv = self._write_patch_csv(
            output.with_name(f"{stem}_training_patches.csv"), training.rgb
        )
        validation_csv = self._write_patch_csv(
            output.with_name(f"{stem}_validation_patches.csv"), validation.rgb
        )
        fitted = fit_calibration_profile(CalibrationProfileRequest(
            training_rgb=training.rgb,
            validation_rgb=validation.rgb,
            training_source=Path(request.training_image),
            validation_source=Path(request.validation_image),
            output_path=output,
            profile_id=request.profile_id,
            camera_id=request.camera_id,
            input_kind=training_kind,
            reference_id=request.reference_id,
            input_domain=request.input_domain,
            white_balance=request.white_balance,
            exposure_normalization=request.exposure_normalization,
            raw_use_camera_wb=request.raw_use_camera_wb,
        ))
        warnings = []
        for label, extraction in (("训练图", training), ("验证图", validation)):
            if extraction.glare_fraction > 0.01:
                warnings.append(
                    f"{label}采样区域有 {extraction.glare_fraction:.1%} 高光像素，请检查反光"
                )
        return CalibrationImageResult(
            profile_path=fitted.profile_path,
            profile_id=fitted.profile_id,
            status=fitted.status,
            selected_model=fitted.selected_model,
            quality=fitted.quality,
            training_preview=training.preview_rgb,
            validation_preview=validation.preview_rgb,
            training_corners=training.corners,
            validation_corners=validation.corners,
            training_patch_csv=training_csv,
            validation_patch_csv=validation_csv,
            warnings=tuple(warnings),
        )
