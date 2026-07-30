#!/usr/bin/env python3
"""Launch the local desktop application."""

from __future__ import annotations

from datetime import datetime
import os
from pathlib import Path
import sys


PROJECT_DIR = Path(__file__).resolve().parent
STARTUP_LOG = PROJECT_DIR / "gui_startup_error.log"


def _report_startup_error(message: str) -> None:
    STARTUP_LOG.write_text(
        f"{datetime.now().isoformat()}\n{message}\n", encoding="utf-8"
    )
    if sys.platform == "win32":
        try:
            import ctypes
            ctypes.windll.user32.MessageBoxW(
                0, message + f"\n\n错误日志：{STARTUP_LOG}",
                "大白菜叶色表型分析 - 启动失败", 0x10,
            )
        except Exception:
            pass


def main() -> int:
    smoke_test = "--smoke-test" in sys.argv
    if smoke_test:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    try:
        from PySide6.QtWidgets import QApplication, QMessageBox
        from ui.main_window import MainWindow
    except ImportError as exc:
        message = (
            "桌面界面依赖尚未安装。请先运行：\n"
            "  python -m pip install -r requirements-gui.txt\n"
            f"原始错误：{exc}"
        )
        print(message, file=sys.stderr)
        _report_startup_error(message)
        return 1
    app = QApplication(sys.argv)
    app.setApplicationName("大白菜叶色表型分析")
    try:
        window = MainWindow(PROJECT_DIR)
        if smoke_test:
            if window.centralWidget().count() != 3:
                raise RuntimeError("桌面界面页面初始化不完整")
            window.close()
            STARTUP_LOG.unlink(missing_ok=True)
            return 0
        window.show()
        STARTUP_LOG.unlink(missing_ok=True)
        return app.exec()
    except Exception as exc:
        message = str(exc)
        _report_startup_error(message)
        QMessageBox.critical(None, "启动失败", message)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
