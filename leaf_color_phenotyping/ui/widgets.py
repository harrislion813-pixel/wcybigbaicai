from __future__ import annotations

from pathlib import Path

import numpy as np
from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QImage, QMouseEvent, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QDialog, QDialogButtonBox, QFileDialog, QHBoxLayout, QLabel, QLineEdit,
    QPushButton, QVBoxLayout, QWidget,
)


def array_to_pixmap(rgb: np.ndarray) -> QPixmap:
    image = np.asarray(rgb)
    if image.dtype != np.uint8:
        image = (np.clip(image, 0, 1) * 255).astype(np.uint8)
    image = np.ascontiguousarray(image)
    height, width = image.shape[:2]
    qimage = QImage(
        image.data, width, height, image.strides[0], QImage.Format.Format_RGB888
    ).copy()
    return QPixmap.fromImage(qimage)


class ImageLabel(QLabel):
    def __init__(self, placeholder: str = "暂无预览"):
        super().__init__(placeholder)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(360, 240)
        self.setStyleSheet("QLabel { background: #18212b; color: #9aa7b2; border-radius: 8px; }")
        self._source_pixmap: QPixmap | None = None

    def set_array(self, rgb: np.ndarray) -> None:
        self._source_pixmap = array_to_pixmap(rgb)
        self._refresh()

    def clear_image(self) -> None:
        self._source_pixmap = None
        self.setPixmap(QPixmap())
        self.setText("暂无预览")

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._refresh()

    def _refresh(self) -> None:
        if self._source_pixmap is not None:
            self.setPixmap(self._source_pixmap.scaled(
                self.size(), Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            ))


class PathSelector(QWidget):
    def __init__(self, mode: str, caption: str, file_filter: str = "所有文件 (*.*)"):
        super().__init__()
        self.mode = mode
        self.caption = caption
        self.file_filter = file_filter
        self.edit = QLineEdit()
        button = QPushButton("浏览…")
        button.clicked.connect(self._browse)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.edit, 1)
        layout.addWidget(button)

    def path(self) -> Path | None:
        value = self.edit.text().strip()
        return Path(value) if value else None

    def set_path(self, path: str | Path) -> None:
        self.edit.setText(str(path))

    def _browse(self) -> None:
        current = self.edit.text().strip()
        if self.mode == "directory":
            selected = QFileDialog.getExistingDirectory(self, self.caption, current)
        elif self.mode == "save":
            selected, _ = QFileDialog.getSaveFileName(
                self, self.caption, current, self.file_filter
            )
        else:
            selected, _ = QFileDialog.getOpenFileName(
                self, self.caption, current, self.file_filter
            )
        if selected:
            self.edit.setText(selected)


class CornerCanvas(QLabel):
    def __init__(self, image_rgb: np.ndarray, corners: np.ndarray | None = None):
        super().__init__()
        self.image_rgb = np.asarray(image_rgb)
        self.source = array_to_pixmap(self.image_rgb)
        self.points: list[QPointF] = []
        if corners is not None:
            self.points = [QPointF(float(x), float(y)) for x, y in corners]
        self.setMinimumSize(760, 500)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)

    def reset_points(self) -> None:
        self.points = []
        self.update()

    def use_full_image(self) -> None:
        height, width = self.image_rgb.shape[:2]
        self.points = [
            QPointF(0, 0), QPointF(width - 1, 0),
            QPointF(width - 1, height - 1), QPointF(0, height - 1),
        ]
        self.update()

    def corners(self) -> np.ndarray | None:
        if len(self.points) != 4:
            return None
        return np.asarray([[point.x(), point.y()] for point in self.points], dtype=np.float32)

    def _draw_geometry(self):
        scaled = self.source.scaled(
            self.size(), Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        offset_x = (self.width() - scaled.width()) / 2
        offset_y = (self.height() - scaled.height()) / 2
        scale_x = scaled.width() / self.source.width()
        scale_y = scaled.height() / self.source.height()
        return scaled, offset_x, offset_y, scale_x, scale_y

    def paintEvent(self, event) -> None:
        super().paintEvent(event)
        scaled, offset_x, offset_y, scale_x, scale_y = self._draw_geometry()
        painter = QPainter(self)
        painter.drawPixmap(int(offset_x), int(offset_y), scaled)
        pen = QPen(Qt.GlobalColor.green, 3)
        painter.setPen(pen)
        mapped = [
            QPointF(offset_x + point.x() * scale_x, offset_y + point.y() * scale_y)
            for point in self.points
        ]
        for index, point in enumerate(mapped):
            painter.drawEllipse(point, 7, 7)
            painter.drawText(point + QPointF(10, -8), str(index + 1))
        if len(mapped) >= 2:
            for index in range(len(mapped) - 1):
                painter.drawLine(mapped[index], mapped[index + 1])
            if len(mapped) == 4:
                painter.drawLine(mapped[-1], mapped[0])

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() != Qt.MouseButton.LeftButton:
            return
        _, offset_x, offset_y, scale_x, scale_y = self._draw_geometry()
        x = (event.position().x() - offset_x) / scale_x
        y = (event.position().y() - offset_y) / scale_y
        height, width = self.image_rgb.shape[:2]
        if not (0 <= x < width and 0 <= y < height):
            return
        point = QPointF(x, y)
        if len(self.points) < 4:
            self.points.append(point)
        else:
            distances = [
                (existing.x() - x) ** 2 + (existing.y() - y) ** 2
                for existing in self.points
            ]
            self.points[int(np.argmin(distances))] = point
        self.update()


class CornerDialog(QDialog):
    def __init__(self, image_rgb: np.ndarray, corners: np.ndarray | None = None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("设置色卡四角")
        self.resize(920, 700)
        instruction = QLabel(
            "依次点击色卡的左上、右上、右下、左下四个角。"
            "色卡可旋转，但灰度色块行必须包含在框内。点击第五次会移动最近的角点。"
        )
        instruction.setWordWrap(True)
        self.canvas = CornerCanvas(image_rgb, corners)
        reset = QPushButton("重新点击")
        reset.clicked.connect(self.canvas.reset_points)
        full = QPushButton("色卡占满整张图")
        full.clicked.connect(self.canvas.use_full_image)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self._accept_if_complete)
        buttons.rejected.connect(self.reject)
        action_row = QHBoxLayout()
        action_row.addWidget(reset)
        action_row.addWidget(full)
        action_row.addStretch(1)
        action_row.addWidget(buttons)
        layout = QVBoxLayout(self)
        layout.addWidget(instruction)
        layout.addWidget(self.canvas, 1)
        layout.addLayout(action_row)

    def _accept_if_complete(self) -> None:
        if self.canvas.corners() is not None:
            self.accept()

    def selected_corners(self) -> np.ndarray | None:
        return self.canvas.corners()
