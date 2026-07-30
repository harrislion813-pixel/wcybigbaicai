from __future__ import annotations

from threading import Event
import traceback

from PySide6.QtCore import QObject, Signal, Slot

from application.analysis_service import AnalysisService
from application.calibration_service import CalibrationService
from application.models import AnalysisRequest, CalibrationImageRequest


class AnalysisWorker(QObject):
    progress = Signal(object)
    succeeded = Signal(object)
    failed = Signal(str)
    finished = Signal()

    def __init__(self, service: AnalysisService, request: AnalysisRequest):
        super().__init__()
        self.service = service
        self.request = request
        self.cancel_event = Event()

    def cancel(self) -> None:
        self.cancel_event.set()

    @Slot()
    def run(self) -> None:
        try:
            result = self.service.run(
                self.request,
                progress_callback=self.progress.emit,
                cancel_event=self.cancel_event,
            )
            self.succeeded.emit(result)
        except Exception:
            self.failed.emit(traceback.format_exc())
        finally:
            self.finished.emit()


class PreviewWorker(QObject):
    succeeded = Signal(object)
    failed = Signal(str)
    finished = Signal()

    def __init__(self, service: AnalysisService, request: AnalysisRequest):
        super().__init__()
        self.service = service
        self.request = request

    @Slot()
    def run(self) -> None:
        try:
            self.succeeded.emit(self.service.preview(self.request))
        except Exception:
            self.failed.emit(traceback.format_exc())
        finally:
            self.finished.emit()


class CalibrationWorker(QObject):
    succeeded = Signal(object)
    failed = Signal(str)
    finished = Signal()

    def __init__(self, service: CalibrationService, request: CalibrationImageRequest):
        super().__init__()
        self.service = service
        self.request = request

    @Slot()
    def run(self) -> None:
        try:
            self.succeeded.emit(self.service.create_profile(self.request))
        except Exception:
            self.failed.emit(traceback.format_exc())
        finally:
            self.finished.emit()

