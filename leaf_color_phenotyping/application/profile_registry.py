from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from src.color_calibration import load_calibration_profile


@dataclass(frozen=True)
class ProfileSummary:
    path: Path
    profile_id: str
    status: str
    camera_id: str
    input_kind: str
    input_domain: str
    white_balance: str
    created_at: str
    median_delta_e: float | None
    error: str | None = None

    @property
    def selectable(self) -> bool:
        return self.status == "validated" and self.error is None


class ProfileRegistry:
    def __init__(self, directories: Iterable[str | Path]):
        self.directories = [Path(item) for item in directories]

    def add_directory(self, directory: str | Path) -> None:
        path = Path(directory)
        if path not in self.directories:
            self.directories.append(path)

    def scan(self) -> list[ProfileSummary]:
        paths: set[Path] = set()
        for directory in self.directories:
            if directory.is_dir():
                paths.update(directory.rglob("*.ccm.json"))
        summaries = [self._summarize(path) for path in sorted(paths)]
        return sorted(summaries, key=lambda item: (not item.selectable, item.profile_id))

    @staticmethod
    def _summarize(path: Path) -> ProfileSummary:
        try:
            profile = load_calibration_profile(path)
            quality = profile.data.get("quality", {})
            median = quality.get("validation_delta_e00", {}).get("median")
            return ProfileSummary(
                path=path.resolve(),
                profile_id=profile.profile_id,
                status=profile.status,
                camera_id=profile.data["input"]["camera_id"],
                input_kind=profile.data["input"]["kind"],
                input_domain=profile.input_domain,
                white_balance=profile.data["preprocessing"]["white_balance"],
                created_at=profile.data["created_at"],
                median_delta_e=float(median) if median is not None else None,
            )
        except Exception as exc:
            return ProfileSummary(
                path=path.resolve(), profile_id=path.stem, status="invalid",
                camera_id="", input_kind="", input_domain="", white_balance="",
                created_at="", median_delta_e=None, error=str(exc),
            )
