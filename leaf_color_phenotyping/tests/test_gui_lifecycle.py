import os
from pathlib import Path
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QObject, Signal, Slot
from PySide6.QtWidgets import QApplication, QMessageBox

from ui.main_window import MainWindow


class SlowWorker(QObject):
    succeeded = Signal(object)
    failed = Signal(str)
    finished = Signal()

    def __init__(self, delay: float = 0.1):
        super().__init__()
        self.delay = delay

    @Slot()
    def run(self) -> None:
        time.sleep(self.delay)
        self.succeeded.emit(None)
        self.finished.emit()


def _process_until(app: QApplication, predicate, timeout: float = 3.0) -> None:
    deadline = time.monotonic() + timeout
    while not predicate() and time.monotonic() < deadline:
        app.processEvents()
        time.sleep(0.01)
    assert predicate()


def test_main_window_defers_close_until_active_worker_finishes(monkeypatch):
    app = QApplication.instance() or QApplication([])
    project_dir = Path(__file__).resolve().parent.parent
    window = MainWindow(project_dir)
    window.show()
    worker = SlowWorker()
    window.analysis_page._start_worker(worker, lambda _: None)
    _process_until(app, window.analysis_page._has_active_jobs)
    monkeypatch.setattr(
        QMessageBox,
        "question",
        lambda *args, **kwargs: QMessageBox.StandardButton.Yes,
    )

    window.close()

    assert window._closing_when_idle
    assert window.isVisible()
    _process_until(app, lambda: not window.isVisible())
    assert window._force_close

