from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QThread, QUrl, Signal, Qt
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QFormLayout, QGridLayout, QGroupBox, QHBoxLayout,
    QFileDialog, QLabel, QLineEdit, QMainWindow, QMessageBox, QProgressBar, QPushButton,
    QSplitter, QTabWidget, QTableWidget, QTableWidgetItem, QTextEdit,
    QVBoxLayout, QWidget,
)

from application.analysis_service import AnalysisService
from application.calibration_service import CalibrationService
from application.models import AnalysisRequest, CalibrationImageRequest
from application.profile_registry import ProfileRegistry

from .widgets import CornerDialog, ImageLabel, PathSelector
from .workers import AnalysisWorker, CalibrationWorker, PreviewWorker


IMAGE_FILTER = "图片 (*.jpg *.jpeg *.png *.tif *.tiff *.bmp *.raw *.dng *.cr2 *.nef *.arw *.raf)"


def _short_error(traceback_text: str) -> str:
    lines = [line.strip() for line in traceback_text.splitlines() if line.strip()]
    return lines[-1] if lines else "未知错误"


class ThreadHost:
    def _init_thread_host(self) -> None:
        self._jobs: list[tuple[QThread, object]] = []

    def _start_worker(self, worker, on_success, on_failure=None) -> None:
        thread = QThread(self)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.succeeded.connect(on_success)
        if on_failure is not None:
            worker.failed.connect(on_failure)
        worker.finished.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        job = (thread, worker)
        self._jobs.append(job)

        def cleanup() -> None:
            if job in self._jobs:
                self._jobs.remove(job)

        thread.finished.connect(cleanup)
        thread.start()


class AnalysisPage(QWidget, ThreadHost):
    def __init__(self, project_dir: Path, registry: ProfileRegistry):
        super().__init__()
        self._init_thread_host()
        self.project_dir = project_dir
        self.registry = registry
        self.service = AnalysisService(project_dir)
        self.preview_items = []
        self.active_worker: AnalysisWorker | None = None

        self.input_path = PathSelector("directory", "选择叶片图片文件夹")
        self.output_path = PathSelector("directory", "选择结果保存文件夹")
        self.output_path.set_path(project_dir / "output")
        self.mode = QComboBox()
        self.mode.addItem("同批次相对比较（不使用 CCM）", "relative")
        self.mode.addItem("跨批次/跨设备比较（必须校准）", "calibrated")
        self.mode.currentIndexChanged.connect(self._mode_changed)
        self.profile = QComboBox()
        self.profile.setEnabled(False)
        self.profile_browse = QPushButton("选择其他…")
        self.profile_browse.setEnabled(False)
        self.profile_browse.clicked.connect(self._browse_profile)
        profile_row = QWidget()
        profile_layout = QHBoxLayout(profile_row)
        profile_layout.setContentsMargins(0, 0, 0, 0)
        profile_layout.addWidget(self.profile, 1)
        profile_layout.addWidget(self.profile_browse)
        self.group_samples = QCheckBox("按样本编号汇总重复图片")
        self.group_samples.setChecked(True)
        self.visualizations = QCheckBox("保存每张图片的分割预览")
        self.visualizations.setChecked(True)
        self.output_format = QComboBox()
        self.output_format.addItem("CSV", "csv")
        self.output_format.addItem("Excel", "excel")
        self.output_format.addItem("JSON", "json")

        basic = QGroupBox("分析任务")
        form = QFormLayout(basic)
        form.addRow("图片文件夹", self.input_path)
        form.addRow("结果文件夹", self.output_path)
        form.addRow("分析模式", self.mode)
        form.addRow("颜色配置", profile_row)
        form.addRow("结果格式", self.output_format)
        form.addRow("", self.group_samples)
        form.addRow("", self.visualizations)

        self.method = QComboBox()
        for text, data in (
            ("自动", "auto"), ("ExG 阈值", "exg"), ("GrabCut", "grabcut"),
            ("U-Net", "unet"), ("SAM", "sam"),
        ):
            self.method.addItem(text, data)
        self.device = QComboBox()
        self.device.addItem("CPU", "cpu")
        self.device.addItem("CUDA", "cuda")
        self.exclude_white = QCheckBox("去除叶柄和主脉等低饱和度白色组织")
        self.exclude_white.setChecked(True)
        self.id_pattern = QLineEdit()
        self.id_pattern.setPlaceholderText("留空时自动从文件名解析")
        advanced = QGroupBox("高级设置")
        advanced.setCheckable(True)
        advanced.setChecked(False)
        advanced_form = QFormLayout(advanced)
        advanced_form.addRow("分割方法", self.method)
        advanced_form.addRow("计算设备", self.device)
        advanced_form.addRow("样本编号正则", self.id_pattern)
        advanced_form.addRow("", self.exclude_white)

        self.preview_button = QPushButton("1. 预检代表图片")
        self.preview_button.clicked.connect(self._preview)
        self.start_button = QPushButton("2. 开始批量分析")
        self.start_button.setProperty("primary", True)
        self.start_button.clicked.connect(self._start_analysis)
        self.cancel_button = QPushButton("取消")
        self.cancel_button.setEnabled(False)
        self.cancel_button.clicked.connect(self._cancel)
        self.open_button = QPushButton("打开结果文件夹")
        self.open_button.setEnabled(False)
        self.open_button.clicked.connect(self._open_output)
        actions = QHBoxLayout()
        for button in (self.preview_button, self.start_button, self.cancel_button, self.open_button):
            actions.addWidget(button)

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumHeight(150)

        self.preview_selector = QComboBox()
        self.preview_selector.currentIndexChanged.connect(self._show_preview)
        self.preview_image = ImageLabel("点击“预检代表图片”查看分割轮廓")
        self.preview_info = QLabel("绿色轮廓应只包围叶片，红色区域表示被排除的白色组织。")
        self.preview_info.setWordWrap(True)
        preview_box = QGroupBox("分割预检")
        preview_layout = QVBoxLayout(preview_box)
        preview_layout.addWidget(self.preview_selector)
        preview_layout.addWidget(self.preview_image, 1)
        preview_layout.addWidget(self.preview_info)

        self.result_table = QTableWidget()
        result_box = QGroupBox("结果预览")
        result_layout = QVBoxLayout(result_box)
        result_layout.addWidget(self.result_table)

        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.addWidget(basic)
        left_layout.addWidget(advanced)
        left_layout.addLayout(actions)
        left_layout.addWidget(self.progress)
        left_layout.addWidget(self.log)
        left_layout.addStretch(1)
        right_split = QSplitter(Qt.Orientation.Vertical)
        right_split.addWidget(preview_box)
        right_split.addWidget(result_box)
        main_split = QSplitter(Qt.Orientation.Horizontal)
        main_split.addWidget(left)
        main_split.addWidget(right_split)
        main_split.setStretchFactor(1, 1)
        layout = QVBoxLayout(self)
        layout.addWidget(main_split)
        self.refresh_profiles()

    def refresh_profiles(self) -> None:
        selected = self.profile.currentData()
        self.profile.clear()
        for summary in self.registry.scan():
            if summary.selectable:
                median = "—" if summary.median_delta_e is None else f"{summary.median_delta_e:.2f}"
                self.profile.addItem(
                    f"{summary.profile_id} · {summary.camera_id} · ΔE中位数 {median}",
                    str(summary.path),
                )
        if selected:
            index = self.profile.findData(selected)
            if index >= 0:
                self.profile.setCurrentIndex(index)

    def _mode_changed(self) -> None:
        calibrated = self.mode.currentData() == "calibrated"
        self.profile.setEnabled(calibrated)
        self.profile_browse.setEnabled(calibrated)

    def _browse_profile(self) -> None:
        selected, _ = QFileDialog.getOpenFileName(
            self, "选择颜色 Profile", str(self.project_dir),
            "CCM Profile (*.ccm.json);;JSON (*.json)",
        )
        if not selected:
            return
        path = Path(selected).resolve()
        self.registry.add_directory(path.parent)
        self.refresh_profiles()
        index = self.profile.findData(str(path))
        if index >= 0:
            self.profile.setCurrentIndex(index)
        else:
            QMessageBox.warning(
                self, "Profile 不可用",
                "该文件不是通过完整性检查的 validated Profile，可在 Profile 管理页查看原因。",
            )

    def _request(self) -> AnalysisRequest:
        input_dir = self.input_path.path()
        output_dir = self.output_path.path()
        if input_dir is None or output_dir is None:
            raise ValueError("请选择图片文件夹和结果文件夹")
        calibrated = self.mode.currentData() == "calibrated"
        profile_data = self.profile.currentData() if calibrated else None
        return AnalysisRequest(
            input_dir=input_dir,
            output_dir=output_dir,
            output_format=self.output_format.currentData(),
            group_by_sample=self.group_samples.isChecked(),
            save_visualizations=self.visualizations.isChecked(),
            calibration_mode=self.mode.currentData(),
            profile_path=Path(profile_data) if profile_data else None,
            segmentation_method=self.method.currentData(),
            device=self.device.currentData(),
            exclude_white_tissue=self.exclude_white.isChecked(),
            id_pattern=self.id_pattern.text().strip() or None,
        )

    def _set_busy(self, busy: bool) -> None:
        self.preview_button.setEnabled(not busy)
        self.start_button.setEnabled(not busy)
        self.cancel_button.setEnabled(busy and self.active_worker is not None)

    def _preview(self) -> None:
        try:
            request = self._request()
        except Exception as exc:
            QMessageBox.warning(self, "设置不完整", str(exc))
            return
        self._set_busy(True)
        self.log.append("正在生成代表图片预检…")
        worker = PreviewWorker(self.service, request)
        self._start_worker(worker, self._preview_ready, self._task_failed)

    def _preview_ready(self, items) -> None:
        self.preview_items = list(items)
        self.preview_selector.clear()
        for item in self.preview_items:
            marker = "⚠" if item.warnings else "✓"
            self.preview_selector.addItem(f"{marker} {item.image_path.name}")
        self._show_preview(0)
        self.log.append(f"预检完成：{len(self.preview_items)} 张代表图片")
        self._set_busy(False)

    def _show_preview(self, index: int) -> None:
        if not 0 <= index < len(self.preview_items):
            return
        item = self.preview_items[index]
        self.preview_image.set_array(item.visualization)
        area = float(item.features.get("QC_mask_area_ratio", 0))
        components = int(item.features.get("QC_component_count", 0))
        warning = "；".join(item.warnings) if item.warnings else "未发现明显异常"
        self.preview_info.setText(
            f"{item.image_path.name}　叶片面积占比 {area:.2%}　"
            f"候选区域 {components}　{warning}"
        )

    def _start_analysis(self) -> None:
        try:
            request = self._request()
        except Exception as exc:
            QMessageBox.warning(self, "设置不完整", str(exc))
            return
        if not self.preview_items:
            answer = QMessageBox.question(
                self, "尚未预检", "尚未检查分割轮廓，仍然开始批量分析吗？"
            )
            if answer != QMessageBox.StandardButton.Yes:
                return
        self.progress.setValue(0)
        self.log.clear()
        worker = AnalysisWorker(self.service, request)
        worker.progress.connect(self._on_progress)
        self.active_worker = worker
        self._set_busy(True)
        self._start_worker(worker, self._analysis_ready, self._task_failed)

    def _on_progress(self, event) -> None:
        if event.total:
            self.progress.setValue(round(event.current * 100 / event.total))
        self.log.append(event.message)

    def _cancel(self) -> None:
        if self.active_worker is not None:
            self.active_worker.cancel()
            self.cancel_button.setEnabled(False)
            self.log.append("已请求取消；当前图片完成后将安全停止。")

    def _analysis_ready(self, result) -> None:
        self.active_worker = None
        self._set_busy(False)
        self.open_button.setEnabled(True)
        self.progress.setValue(100 if not result.cancelled else self.progress.value())
        dataframe = result.dataframe
        max_rows, max_columns = 200, 60
        rows = min(len(dataframe), max_rows)
        columns = list(dataframe.columns[:max_columns])
        self.result_table.setRowCount(rows)
        self.result_table.setColumnCount(len(columns))
        self.result_table.setHorizontalHeaderLabels([str(item) for item in columns])
        for row in range(rows):
            for column, name in enumerate(columns):
                self.result_table.setItem(
                    row, column, QTableWidgetItem(str(dataframe.iloc[row][name]))
                )
        self.result_table.resizeColumnsToContents()
        state = "已取消并保存部分结果" if result.cancelled else "分析完成"
        self.log.append(
            f"{state}：{len(dataframe)} 行结果，{len(result.failures)} 张图片失败。"
        )
        QMessageBox.information(self, state, f"结果位置：\n{result.table_path}")

    def _task_failed(self, details: str) -> None:
        self.active_worker = None
        self._set_busy(False)
        self.log.append(details)
        QMessageBox.critical(self, "任务失败", _short_error(details))

    def _open_output(self) -> None:
        path = self.output_path.path()
        if path is not None:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(path.resolve())))


class CalibrationPage(QWidget, ThreadHost):
    profile_created = Signal(str)

    def __init__(self, project_dir: Path):
        super().__init__()
        self._init_thread_host()
        self.project_dir = project_dir
        self.service = CalibrationService()
        self.training_corners = None
        self.validation_corners = None

        self.training = PathSelector("file", "选择训练色卡图", IMAGE_FILTER)
        self.validation = PathSelector("file", "选择独立验证色卡图", IMAGE_FILTER)
        self.output = PathSelector("save", "保存颜色 Profile", "CCM Profile (*.ccm.json)")
        self.output.set_path(project_dir / "models" / "camera_profile.ccm.json")
        self.profile_id = QLineEdit("camera-profile-v1")
        self.camera_id = QLineEdit()
        self.camera_id.setPlaceholderText("例如 camera-a；同一相机和拍摄方案保持一致")
        self.reference = QComboBox()
        self.reference.addItem("2014 年 11 月后版本", "after_nov_2014")
        self.reference.addItem("2014 年 11 月前版本", "before_nov_2014")
        self.raw_camera_wb = QCheckBox("RAW 使用相机记录的白平衡")
        self.raw_camera_wb.setChecked(True)

        form_box = QGroupBox("颜色校准向导")
        form = QFormLayout(form_box)
        form.addRow("训练色卡图", self.training)
        form.addRow("独立验证图", self.validation)
        form.addRow("Profile 名称", self.profile_id)
        form.addRow("相机/拍摄方案 ID", self.camera_id)
        form.addRow("色卡版本", self.reference)
        form.addRow("保存位置", self.output)
        form.addRow("", self.raw_camera_wb)

        train_corner = QPushButton("手工设置训练图四角")
        train_corner.clicked.connect(lambda: self._edit_corners("training"))
        validation_corner = QPushButton("手工设置验证图四角")
        validation_corner.clicked.connect(lambda: self._edit_corners("validation"))
        self.create_button = QPushButton("自动识别色块并创建 Profile")
        self.create_button.setProperty("primary", True)
        self.create_button.clicked.connect(self._create)
        actions = QHBoxLayout()
        actions.addWidget(train_corner)
        actions.addWidget(validation_corner)
        actions.addStretch(1)
        actions.addWidget(self.create_button)

        note = QLabel(
            "训练图和验证图必须是两次独立拍摄，并保持相机、镜头、光源、曝光和白平衡一致。"
            "自动识别失败时再使用手工四角；普通用户无需填写 RGB 数值尺度或矩阵参数。"
        )
        note.setWordWrap(True)
        note.setStyleSheet("QLabel { padding: 10px; background: #fff5d6; border-radius: 6px; }")

        self.training_preview = ImageLabel("训练图色块预览")
        self.validation_preview = ImageLabel("验证图色块预览")
        previews = QSplitter(Qt.Orientation.Horizontal)
        previews.addWidget(self.training_preview)
        previews.addWidget(self.validation_preview)
        self.result = QTextEdit()
        self.result.setReadOnly(True)
        self.result.setMaximumHeight(180)

        layout = QVBoxLayout(self)
        layout.addWidget(form_box)
        layout.addWidget(note)
        layout.addLayout(actions)
        layout.addWidget(previews, 1)
        layout.addWidget(self.result)

    def _edit_corners(self, which: str) -> None:
        selector = self.training if which == "training" else self.validation
        path = selector.path()
        if path is None:
            QMessageBox.warning(self, "缺少图片", "请先选择色卡图片")
            return
        try:
            display = self.service.load_display_image(path)
        except Exception as exc:
            QMessageBox.critical(self, "无法读取图片", str(exc))
            return
        existing = self.training_corners if which == "training" else self.validation_corners
        dialog = CornerDialog(display, existing, self)
        if dialog.exec():
            if which == "training":
                self.training_corners = dialog.selected_corners()
            else:
                self.validation_corners = dialog.selected_corners()

    def _request(self) -> CalibrationImageRequest:
        training = self.training.path()
        validation = self.validation.path()
        output = self.output.path()
        if training is None or validation is None or output is None:
            raise ValueError("请选择训练图、独立验证图和 Profile 保存位置")
        if not str(output).lower().endswith(".ccm.json"):
            output = Path(str(output) + ".ccm.json")
            self.output.set_path(output)
        return CalibrationImageRequest(
            training_image=training,
            validation_image=validation,
            output_path=output,
            profile_id=self.profile_id.text().strip(),
            camera_id=self.camera_id.text().strip(),
            reference_id=self.reference.currentData(),
            raw_use_camera_wb=self.raw_camera_wb.isChecked(),
            training_corners=self.training_corners,
            validation_corners=self.validation_corners,
        )

    def _create(self) -> None:
        try:
            request = self._request()
        except Exception as exc:
            QMessageBox.warning(self, "设置不完整", str(exc))
            return
        self.create_button.setEnabled(False)
        self.result.setText("正在读取色卡、提取 24 个色块并进行独立验证…")
        worker = CalibrationWorker(self.service, request)
        self._start_worker(worker, self._ready, self._failed)

    def _ready(self, result) -> None:
        self.create_button.setEnabled(True)
        self.training_corners = result.training_corners
        self.validation_corners = result.validation_corners
        self.training_preview.set_array(result.training_preview)
        self.validation_preview.set_array(result.validation_preview)
        metrics = result.quality["validation_delta_e00"]
        failures = result.quality.get("failures", [])
        status_text = "验证通过，可以用于跨批次分析" if result.status == "validated" else "未通过，只保存为 draft"
        lines = [
            f"状态：{status_text}",
            f"模型：{result.selected_model}",
            f"验证 ΔE00：中位数 {metrics['median']:.3f}，P95 {metrics['p95']:.3f}，最大值 {metrics['max']:.3f}",
            f"Profile：{result.profile_path}",
            f"取样数据：{result.training_patch_csv.name} / {result.validation_patch_csv.name}",
        ]
        if failures:
            lines.append("未通过指标：" + "；".join(failures))
        lines.extend(result.warnings)
        self.result.setText("\n".join(lines))
        self.profile_created.emit(str(result.profile_path))
        QMessageBox.information(self, "颜色校准完成", status_text)

    def _failed(self, details: str) -> None:
        self.create_button.setEnabled(True)
        self.result.setText(details)
        QMessageBox.critical(self, "颜色校准失败", _short_error(details))


class ProfilePage(QWidget):
    def __init__(self, registry: ProfileRegistry):
        super().__init__()
        self.registry = registry
        self.table = QTableWidget()
        refresh = QPushButton("刷新")
        refresh.clicked.connect(self.refresh)
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("只有状态为 validated 且完整性检查通过的 Profile 才能用于跨批次分析。"))
        layout.addWidget(refresh, alignment=Qt.AlignmentFlag.AlignLeft)
        layout.addWidget(self.table)
        self.refresh()

    def refresh(self) -> None:
        summaries = self.registry.scan()
        headers = ["状态", "Profile", "相机", "图片类型", "工作域", "白平衡", "ΔE中位数", "文件", "问题"]
        self.table.setColumnCount(len(headers))
        self.table.setHorizontalHeaderLabels(headers)
        self.table.setRowCount(len(summaries))
        for row, item in enumerate(summaries):
            values = [
                item.status, item.profile_id, item.camera_id, item.input_kind,
                item.input_domain, item.white_balance,
                "" if item.median_delta_e is None else f"{item.median_delta_e:.3f}",
                str(item.path), item.error or "",
            ]
            for column, value in enumerate(values):
                self.table.setItem(row, column, QTableWidgetItem(value))
        self.table.resizeColumnsToContents()


class MainWindow(QMainWindow):
    def __init__(self, project_dir: Path | None = None):
        super().__init__()
        self.project_dir = Path(project_dir or Path(__file__).resolve().parent.parent)
        self.setWindowTitle("大白菜叶色表型分析")
        self.resize(1420, 900)
        registry = ProfileRegistry([self.project_dir / "models"])
        self.analysis_page = AnalysisPage(self.project_dir, registry)
        self.calibration_page = CalibrationPage(self.project_dir)
        self.profile_page = ProfilePage(registry)
        self.calibration_page.profile_created.connect(self._profile_created)
        tabs = QTabWidget()
        tabs.addTab(self.analysis_page, "叶片分析")
        tabs.addTab(self.calibration_page, "颜色校准")
        tabs.addTab(self.profile_page, "Profile 管理")
        self.setCentralWidget(tabs)
        self.setStyleSheet("""
            QWidget { font-size: 13px; }
            QGroupBox { font-weight: 600; margin-top: 12px; padding-top: 10px; }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 4px; }
            QPushButton { padding: 7px 14px; }
            QPushButton[primary="true"] { background: #217346; color: white; font-weight: 600; }
            QLineEdit, QComboBox { min-height: 28px; }
        """)

    def _profile_created(self, profile_path: str) -> None:
        self.profile_page.registry.add_directory(Path(profile_path).parent)
        self.analysis_page.refresh_profiles()
        self.profile_page.refresh()
