#!/usr/bin/env python3
"""Launch the local desktop application."""

from __future__ import annotations

from pathlib import Path
import sys


def main() -> int:
    try:
        from PySide6.QtWidgets import QApplication, QMessageBox
        from ui.main_window import MainWindow
    except ImportError as exc:
        print(
            "桌面界面依赖尚未安装。请先运行：\n"
            "  python -m pip install -r requirements-gui.txt\n"
            f"原始错误：{exc}",
            file=sys.stderr,
        )
        return 1
    app = QApplication(sys.argv)
    app.setApplicationName("大白菜叶色表型分析")
    try:
        window = MainWindow(Path(__file__).resolve().parent)
        window.show()
        return app.exec()
    except Exception as exc:
        QMessageBox.critical(None, "启动失败", str(exc))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

