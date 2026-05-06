import math
import os

from PIL import Image
from PyQt6.QtCore import QSettings, QSignalBlocker, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QImage, QPixmap
from PyQt6.QtWidgets import (
    QColorDialog,
    QComboBox,
    QFileDialog,
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QProgressBar,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from core.collage_batch_runner import CollageBatchRunner
from core.collage_processor import calculate_auto_split, create_collage
from models.collage_model import CollageManager, CollageTemplate
from ui.diversify_widget import DiversifyWidget


_GREEN = "#07C160"
_SIDE = "#EFEFEF"
_CARD = "#FFFFFF"
_INPUT = "#F0F0F0"
_SEP = "#E5E5E5"
_TEXT = "#191919"
_TEXT2 = "#888888"
_RED = "#FA5151"

_PRESETS = ("2×2", "2×3", "3×4", "2×4", "4×4", "1×3", "3×1")
_ASPECTS = {
    "16:9": 16 / 9,
    "4:3": 4 / 3,
    "3:4": 3 / 4,
    "1:1": 1,
    "自适应": 0,
}


class CollageTab(QWidget):
    """拼图 Tab 主容器（本阶段只实现左侧 sidebar，右侧留 placeholder）。"""

    config_changed = pyqtSignal(object)

    def __init__(self, collages_dir: str, parent=None):
        super().__init__(parent)
        self._mgr = CollageManager(collages_dir)
        self._current_collage: CollageTemplate | None = None
        self._preset_buttons: dict[str, QPushButton] = {}
        self._preview_cells: list[QLabel] = []
        self._suppress_emit = False
        self._image_files: list[str] = []
        self._input_dir = ""
        self._output_dir = ""
        self._excluded_indices: set[int] = set()
        self._show_all_thumbs = False
        self._preview_collage_index = 0
        self._collage_runner: CollageBatchRunner | None = None

        self._build_ui()
        self._load_template_list()
        self._wire_signals()
        self._refresh_preset_state()
        self._refresh_preview()

    # ── Public API ────────────────────────────────────────────────
    def get_current_config(self) -> CollageTemplate:
        """返回当前 UI 表单中的 CollageTemplate（不要求已保存）。"""
        name = self._current_collage.name if self._current_collage else "未命名拼图"
        return CollageTemplate(
            name=name,
            layout="grid",
            rows=self._row_spin.value(),
            cols=self._col_spin.value(),
            gap=self._gap_spin.value(),
            padding=self._padding_spin.value(),
            background_color=self._background_edit.text().strip() or "#FFFFFF",
            cell_aspect_ratio=self._current_aspect_ratio(),
            output_width=1920,
            output_height=0,
        )

    def set_config(self, tpl: CollageTemplate):
        """从外部设置（不触发 config_changed）。"""
        self._suppress_emit = True
        blockers = [
            QSignalBlocker(self._row_spin),
            QSignalBlocker(self._col_spin),
            QSignalBlocker(self._gap_spin),
            QSignalBlocker(self._padding_spin),
            QSignalBlocker(self._aspect_combo),
            QSignalBlocker(self._background_edit),
        ]
        self._current_collage = tpl
        self._row_spin.setValue(tpl.rows)
        self._col_spin.setValue(tpl.cols)
        self._gap_spin.setValue(tpl.gap)
        self._padding_spin.setValue(tpl.padding)
        self._aspect_combo.setCurrentText(self._aspect_label_for(tpl.cell_aspect_ratio))
        self._background_edit.setText(tpl.background_color)
        self._update_color_swatch()
        self._refresh_preset_state()
        self._refresh_preview()
        del blockers
        self._suppress_emit = False
        self._refresh_right_state()

    def get_diversify_config(self):
        """返回嵌入的 DiversifyWidget 的当前配置。"""
        return self._diversify.get_config()

    # ── UI ────────────────────────────────────────────────────────
    def _build_ui(self):
        self.setObjectName("CollageTab")
        self.setStyleSheet(
            f"""
            QWidget#collage_sidebar,
            QWidget#collage_scroll_body {{
                background: {_SIDE};
            }}
            QWidget#collage_sidebar {{
                border-right: 1px solid {_SEP};
            }}
            QWidget#CollageTab QLabel#h2 {{
                color: {_TEXT};
                font-size: 15px;
                font-weight: 700;
                border-left: 3px solid {_GREEN};
                padding-left: 8px;
            }}
            QWidget#CollageTab QLabel#cap {{
                color: {_TEXT2};
                font-size: 11px;
                font-weight: 500;
            }}
            QWidget#CollageTab QLabel#hint {{
                color: {_TEXT2};
                font-size: 12px;
            }}
            QWidget#CollageTab QWidget#tpl_list_frame,
            QWidget#CollageTab QWidget#mini_grid_frame {{
                background: {_CARD};
                border: 1px solid {_SEP};
                border-radius: 10px;
            }}
            QWidget#CollageTab QPushButton#preset_btn {{
                background: {_INPUT};
                color: {_TEXT};
                border: 1px solid {_SEP};
                border-radius: 8px;
                padding: 7px 10px;
                font-weight: 600;
            }}
            QWidget#CollageTab QPushButton#preset_btn:checked {{
                background: {_GREEN};
                color: white;
                border-color: {_GREEN};
            }}
            QWidget#CollageTab QPushButton#danger {{
                color: {_RED};
            }}
            QWidget#CollageTab QLabel#color_swatch {{
                border: 1px solid {_SEP};
                border-radius: 6px;
            }}
            QWidget#CollageTab QLabel#mini_cell {{
                background: {_INPUT};
                color: {_TEXT2};
                border: 1px solid {_SEP};
                border-radius: 4px;
                font-size: 11px;
            }}
            """
        )

        main = QHBoxLayout(self)
        main.setContentsMargins(0, 0, 0, 0)
        main.setSpacing(0)
        sidebar = self._build_sidebar()
        main.addWidget(sidebar, 0)

        right_scroll = QScrollArea()
        right_scroll.setWidgetResizable(True)
        right_scroll.setFrameShape(QFrame.Shape.NoFrame)
        right_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        right_scroll.setWidget(self._build_right_steps())
        main.addWidget(right_scroll, 1)

    def _build_sidebar(self) -> QWidget:
        sidebar = QWidget()
        sidebar.setObjectName("collage_sidebar")
        sidebar.setFixedWidth(380)
        layout = QVBoxLayout(sidebar)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        body = QWidget()
        body.setObjectName("collage_scroll_body")
        content = QVBoxLayout(body)
        content.setContentsMargins(16, 18, 16, 18)
        content.setSpacing(10)

        self._add_layout_section(content)
        content.addWidget(self._sep())
        self._add_template_section(content)
        content.addWidget(self._sep())
        content.addWidget(self._label("批次差异化", "h2"))
        self._diversify = DiversifyWidget(self)
        content.addWidget(self._diversify)
        content.addStretch(1)

        scroll.setWidget(body)
        layout.addWidget(scroll)
        return sidebar

    def _add_layout_section(self, content: QVBoxLayout):
        content.addWidget(self._label("拼图布局", "h2"))
        content.addWidget(self._label("快捷预设", "cap"))

        preset_row = QHBoxLayout()
        preset_row.setSpacing(6)
        for name in _PRESETS:
            btn = QPushButton(name)
            btn.setObjectName("preset_btn")
            btn.setCheckable(True)
            btn.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            preset_row.addWidget(btn)
            self._preset_buttons[name] = btn
        content.addLayout(preset_row)

        grid = QGridLayout()
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(8)
        self._row_spin = self._spin(1, 4, 3)
        self._col_spin = self._spin(1, 6, 4)
        self._gap_spin = self._spin(0, 20, 4)
        self._padding_spin = self._spin(0, 40, 0)
        self._add_grid_field(grid, 0, 0, "行数", self._row_spin)
        self._add_grid_field(grid, 0, 1, "列数", self._col_spin)
        self._add_grid_field(grid, 1, 0, "间距 (px)", self._gap_spin)
        self._add_grid_field(grid, 1, 1, "边距 (px)", self._padding_spin)
        content.addLayout(grid)

        lower_grid = QGridLayout()
        lower_grid.setHorizontalSpacing(10)
        lower_grid.setVerticalSpacing(8)
        self._aspect_combo = QComboBox()
        self._aspect_combo.addItems(_ASPECTS.keys())
        self._background_edit = QLineEdit()
        self._background_edit.setMaxLength(7)
        self._background_edit.setText(self._last_background_color())
        self._color_swatch = QLabel()
        self._color_swatch.setObjectName("color_swatch")
        self._color_swatch.setFixedSize(28, 28)
        self._color_swatch.setCursor(Qt.CursorShape.PointingHandCursor)
        self._color_swatch.mousePressEvent = lambda event: self._choose_background_color()
        bg_row = QHBoxLayout()
        bg_row.setSpacing(6)
        bg_row.addWidget(self._background_edit, 1)
        bg_row.addWidget(self._color_swatch, 0)
        self._add_grid_field(lower_grid, 0, 0, "单元格比例", self._aspect_combo)
        self._add_grid_field(lower_grid, 0, 1, "背景色", bg_row)
        content.addLayout(lower_grid)
        self._update_color_swatch()

        self._preview_label = self._label("", "cap")
        content.addWidget(self._preview_label)
        self._mini_grid_frame = QWidget()
        self._mini_grid_frame.setObjectName("mini_grid_frame")
        self._mini_grid = QGridLayout(self._mini_grid_frame)
        self._mini_grid.setContentsMargins(8, 8, 8, 8)
        self._mini_grid.setSpacing(5)
        content.addWidget(self._mini_grid_frame)

    def _add_template_section(self, content: QVBoxLayout):
        content.addWidget(self._label("拼图模板", "h2"))
        tpl_frame = QWidget()
        tpl_frame.setObjectName("tpl_list_frame")
        frame_layout = QVBoxLayout(tpl_frame)
        frame_layout.setContentsMargins(4, 4, 4, 4)
        self._template_list = QListWidget()
        self._template_list.setMinimumHeight(100)
        self._template_list.setMaximumHeight(180)
        frame_layout.addWidget(self._template_list)
        content.addWidget(tpl_frame)

        row = QHBoxLayout()
        row.setSpacing(8)
        self._save_template_btn = QPushButton("保存模板")
        self._delete_template_btn = QPushButton("删除")
        self._delete_template_btn.setObjectName("danger")
        row.addWidget(self._save_template_btn)
        row.addWidget(self._delete_template_btn)
        content.addLayout(row)

    def _wire_signals(self):
        for name, btn in self._preset_buttons.items():
            btn.clicked.connect(lambda checked=False, n=name: self._on_preset_clicked(n))
        for spin in (self._row_spin, self._col_spin, self._gap_spin, self._padding_spin):
            spin.valueChanged.connect(self._on_form_changed)
        self._aspect_combo.currentTextChanged.connect(self._on_form_changed)
        self._background_edit.textChanged.connect(self._on_background_changed)
        self._template_list.itemClicked.connect(self._on_template_item_clicked)
        self._save_template_btn.clicked.connect(self._save_template_to_disk)
        self._delete_template_btn.clicked.connect(self._delete_selected_template)
        self._diversify.config_changed.connect(lambda _cfg: self._emit_config_changed())
        self.config_changed.connect(lambda _cfg: self._refresh_right_state())

    def _build_right_steps(self) -> QWidget:
        body = QWidget()
        body.setObjectName("collage_scroll_body")
        layout = QVBoxLayout(body)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(14)
        layout.addWidget(self._build_step1_card())
        layout.addWidget(self._build_step2_card())
        layout.addWidget(self._build_step3_card())
        layout.addWidget(self._build_step4_card())
        layout.addStretch(1)
        return body

    def _build_step1_card(self) -> QWidget:
        card, layout = self._step_card("1", "选择图片")
        row = QHBoxLayout()
        row.setSpacing(12)
        btn = QPushButton("选择图片文件夹")
        btn.clicked.connect(self._choose_input_dir)
        self._input_path_label = self._label("未选择", "hint")
        self._input_count_label = self._label("已扫描到 0 张图片", "hint")
        texts = QVBoxLayout()
        texts.setSpacing(3)
        texts.addWidget(self._input_path_label)
        texts.addWidget(self._input_count_label)
        row.addWidget(btn, 0)
        row.addLayout(texts, 1)
        layout.addLayout(row)
        last = QSettings("融景", "RongJing").value("collage/last_input_dir", "", type=str)
        if last:
            self._input_path_label.setText(last)
        return card

    def _build_step2_card(self) -> QWidget:
        card, layout = self._step_card("2", "选择页面 & 自动拆分")
        summary = QFrame()
        summary.setStyleSheet(f"QFrame {{ background: #F8F8F8; border: 1px solid {_SEP}; border-radius: 8px; }}")
        row = QHBoxLayout(summary)
        row.setContentsMargins(12, 10, 12, 10)
        row.setSpacing(16)
        self._selected_pages_label = self._label("已选 0/0 页", "cap")
        self._output_count_spin = self._spin(1, 1, 1)
        self._output_count_spin.valueChanged.connect(self._on_output_count_changed)
        self._pages_per_label = self._label("每张 0 页", "cap")
        row.addWidget(self._selected_pages_label)
        row.addWidget(self._label("想要的输出图片数", "cap"))
        row.addWidget(self._output_count_spin)
        row.addWidget(self._pages_per_label)
        row.addStretch(1)
        layout.addWidget(summary)
        layout.addWidget(self._label("点击缩略图排除不需要的页面（红色 = 已排除）", "hint"))
        self._thumb_grid = QGridLayout()
        self._thumb_grid.setSpacing(8)
        layout.addLayout(self._thumb_grid)
        self._show_all_btn = QPushButton("显示全部")
        self._show_all_btn.clicked.connect(self._show_all_thumbnails)
        layout.addWidget(self._show_all_btn, 0, Qt.AlignmentFlag.AlignCenter)
        return card

    def _build_step3_card(self) -> QWidget:
        card, layout = self._step_card("3", "拼图预览")
        row = QHBoxLayout()
        row.setSpacing(8)
        self._preview_first_btn = QPushButton("第 1 张")
        self._preview_second_btn = QPushButton("第 2 张")
        self._preview_first_btn.clicked.connect(lambda: self._set_preview_index(0))
        self._preview_second_btn.clicked.connect(lambda: self._set_preview_index(1))
        row.addWidget(self._preview_first_btn)
        row.addWidget(self._preview_second_btn)
        row.addStretch(1)
        layout.addLayout(row)
        self._collage_preview_label = QLabel("请选择图片文件夹")
        self._collage_preview_label.setObjectName("hint")
        self._collage_preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._collage_preview_label.setMinimumHeight(260)
        self._collage_preview_label.setStyleSheet(f"background: {_CARD}; border: 1px solid {_SEP}; border-radius: 8px;")
        layout.addWidget(self._collage_preview_label)
        self._preview_hint_label = self._label("第 1 张拼图", "hint")
        self._preview_hint_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._preview_hint_label)
        return card

    def _build_step4_card(self) -> QWidget:
        card, layout = self._step_card("4", "导出")
        row = QHBoxLayout()
        row.setSpacing(12)
        btn = QPushButton("选择输出文件夹")
        btn.clicked.connect(self._choose_output_dir)
        self._output_path_label = self._label("未选择", "hint")
        self._format_combo = QComboBox()
        self._format_combo.addItems(["PNG", "JPEG"])
        row.addWidget(btn, 0)
        row.addWidget(self._output_path_label, 1)
        row.addWidget(self._format_combo, 0)
        layout.addLayout(row)
        self._run_collage_btn = QPushButton("开始拼图（1 张）")
        self._run_collage_btn.setStyleSheet(f"QPushButton {{ background: {_GREEN}; color: white; border: 0; border-radius: 8px; padding: 12px; font-weight: 700; }}")
        self._run_collage_btn.clicked.connect(lambda: self._run_collage_batch())
        layout.addWidget(self._run_collage_btn)
        self._collage_progress = QProgressBar()
        self._collage_progress.setVisible(False)
        layout.addWidget(self._collage_progress)
        self._collage_status = self._label("", "hint")
        self._collage_status.setVisible(False)
        layout.addWidget(self._collage_status)
        self._cancel_collage_btn = QPushButton("取消")
        self._cancel_collage_btn.setVisible(False)
        self._cancel_collage_btn.clicked.connect(self._abort_collage_batch)
        layout.addWidget(self._cancel_collage_btn, 0, Qt.AlignmentFlag.AlignRight)
        last = QSettings("融景", "RongJing").value("collage/last_output_dir", "", type=str)
        if last:
            self._output_dir = last
            self._output_path_label.setText(last)
        return card

    def _choose_input_dir(self):
        path = QFileDialog.getExistingDirectory(self, "选择图片文件夹", self._input_dir)
        if path:
            self._set_input_dir(path)

    def _choose_output_dir(self):
        path = QFileDialog.getExistingDirectory(self, "选择输出文件夹", self._output_dir)
        if path:
            self._set_output_dir(path)

    def _set_input_dir(self, path: str):
        self._input_dir = path
        QSettings("融景", "RongJing").setValue("collage/last_input_dir", path)
        self._image_files = self._scan_image_files(path)
        self._excluded_indices = set()
        self._show_all_thumbs = False
        self._preview_collage_index = 0
        self._input_path_label.setText(path)
        self._input_count_label.setText(f"已扫描到 {len(self._image_files)} 张图片")
        self._reset_output_count_for_selection()
        self._refresh_thumbnails()
        self._refresh_right_state()

    def _set_output_dir(self, path: str):
        self._output_dir = path
        QSettings("融景", "RongJing").setValue("collage/last_output_dir", path)
        self._output_path_label.setText(path)

    def _toggle_excluded(self, index: int):
        if index < 0 or index >= len(self._image_files):
            return
        if index in self._excluded_indices:
            self._excluded_indices.remove(index)
        else:
            self._excluded_indices.add(index)
        self._refresh_thumbnails()
        self._refresh_right_state()

    def _on_output_count_changed(self, *_args):
        self._refresh_right_state()

    def _show_all_thumbnails(self):
        self._show_all_thumbs = True
        self._refresh_thumbnails()

    def _set_preview_index(self, index: int):
        self._preview_collage_index = index
        self._refresh_collage_preview()

    def _run_collage_batch(self, callback=None):
        if not self._image_files:
            QMessageBox.warning(self, "提示", "请先选择图片文件夹")
            return
        if not self._output_dir:
            QMessageBox.warning(self, "提示", "请先选择输出文件夹")
            return
        cfg = self.get_current_config()
        diversify_cfg = self._diversify.get_config()
        self._collage_runner = CollageBatchRunner(
            self._image_files,
            cfg,
            self._output_dir,
            self._format_combo.currentText(),
            self._output_count_spin.value(),
            set(self._excluded_indices),
            diversify_cfg,
            self,
        )
        self._collage_runner.progress.connect(self._on_collage_progress)
        self._collage_runner.finished.connect(self._on_collage_finished)
        if callback is not None:
            self._collage_runner.finished.connect(callback)
        self._run_collage_btn.setVisible(False)
        self._cancel_collage_btn.setVisible(True)
        self._collage_progress.setVisible(True)
        self._collage_status.setVisible(True)
        self._collage_progress.setValue(0)
        self._collage_status.setText("正在拼图…")
        self._collage_runner.start()

    def _abort_collage_batch(self):
        if self._collage_runner is not None:
            self._collage_runner.abort()

    def _on_collage_progress(self, done: int, total: int, msg: str):
        self._collage_progress.setMaximum(max(1, total))
        self._collage_progress.setValue(done)
        self._collage_status.setText(f"[{done} / {total}] {msg}")

    def _on_collage_finished(self, success: bool, msg: str):
        self._run_collage_btn.setVisible(True)
        self._cancel_collage_btn.setVisible(False)
        self._collage_status.setText(msg)
        if os.environ.get("QT_QPA_PLATFORM") != "offscreen":
            if success:
                QMessageBox.information(self, "完成", msg)
            else:
                QMessageBox.warning(self, "处理结果", msg)

    def _refresh_right_state(self):
        if not hasattr(self, "_output_count_spin"):
            return
        selected = self._selected_image_files()
        total = len(self._image_files)
        selected_count = len(selected)
        max_outputs = max(1, selected_count)
        if self._output_count_spin.maximum() != max_outputs:
            self._output_count_spin.setMaximum(max_outputs)
        if self._output_count_spin.value() > max_outputs:
            self._output_count_spin.setValue(max_outputs)
        ranges = self._current_ranges()
        pages_per = max((end - start for start, end in ranges), default=0)
        self._selected_pages_label.setText(f"已选 {selected_count}/{total} 页")
        self._pages_per_label.setText(f"每张 {pages_per} 页")
        self._run_collage_btn.setText(f"开始拼图（{len(ranges)} 张）")
        self._preview_first_btn.setEnabled(len(ranges) >= 1)
        self._preview_second_btn.setEnabled(len(ranges) >= 2)
        self._refresh_collage_preview()

    def _refresh_thumbnails(self):
        while self._thumb_grid.count():
            item = self._thumb_grid.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        limit = len(self._image_files) if self._show_all_thumbs else min(24, len(self._image_files))
        for index, path in enumerate(self._image_files[:limit]):
            self._thumb_grid.addWidget(self._thumbnail_widget(index, path), index // 6, index % 6)
        hidden = len(self._image_files) - limit
        self._show_all_btn.setVisible(hidden > 0)
        if hidden > 0:
            self._show_all_btn.setText(f"显示全部（还有 {hidden} 张）")

    def _refresh_collage_preview(self):
        ranges = self._current_ranges()
        if not ranges:
            self._collage_preview_label.setPixmap(QPixmap())
            self._collage_preview_label.setText("请选择图片文件夹")
            self._preview_hint_label.setText("无可预览图片")
            return
        self._preview_collage_index = min(self._preview_collage_index, len(ranges) - 1)
        selected = self._selected_image_files()
        start, end = ranges[self._preview_collage_index]
        images = []
        try:
            for path in selected[start:end]:
                with Image.open(path) as img:
                    images.append(img.copy())
            cfg = self.get_current_config()
            preview = create_collage(
                images,
                cfg.layout,
                cfg.rows,
                cfg.cols,
                gap=cfg.gap,
                padding=cfg.padding,
                background_color=cfg.background_color,
                cell_aspect_ratio=cfg.cell_aspect_ratio,
                output_width=900,
                output_height=0,
            )
            preview.thumbnail((600, 600), Image.LANCZOS)
            self._collage_preview_label.setText("")
            self._collage_preview_label.setPixmap(self._pil_to_pixmap(preview))
            self._preview_hint_label.setText(
                f"第 {self._preview_collage_index + 1} 张拼图（{end - start} 页）· 共 {len(ranges)} 张"
            )
        except Exception as exc:
            self._collage_preview_label.setPixmap(QPixmap())
            self._collage_preview_label.setText(f"预览失败：{exc}")

    def _reset_output_count_for_selection(self):
        selected_count = len(self._selected_image_files())
        cells = max(1, self.get_current_config().total_cells)
        value = max(1, math.ceil(selected_count / cells)) if selected_count else 1
        self._output_count_spin.setMaximum(max(1, selected_count))
        self._output_count_spin.setValue(value)

    def _selected_image_files(self) -> list[str]:
        return [path for idx, path in enumerate(self._image_files) if idx not in self._excluded_indices]

    def _current_ranges(self):
        return calculate_auto_split(
            len(self._selected_image_files()),
            self._output_count_spin.value(),
            max(1, self.get_current_config().total_cells),
        )

    def _thumbnail_widget(self, index: int, path: str) -> QWidget:
        tile = QWidget()
        tile.setObjectName("thumb_tile")
        tile.setFixedSize(92, 108)
        excluded = index in self._excluded_indices
        border = _RED if excluded else _GREEN
        tile.setStyleSheet(f"QWidget#thumb_tile {{ background: white; border: 2px solid {border}; border-radius: 8px; }}")
        layout = QVBoxLayout(tile)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(3)
        image_label = QLabel()
        image_label.setFixedSize(80, 80)
        image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        image_label.setPixmap(self._load_thumb_pixmap(path))
        mark = QLabel("×" if excluded else "✓", tile)
        mark.setAlignment(Qt.AlignmentFlag.AlignCenter)
        mark.setStyleSheet(f"background: {border}; color: white; border-radius: 9px; font-weight: 700;")
        mark.setFixedSize(18, 18)
        mark.move(68, 4)
        num = QLabel(f"P{index + 1}")
        num.setAlignment(Qt.AlignmentFlag.AlignCenter)
        num.setStyleSheet("border: 0; color: #666; font-size: 11px;")
        layout.addWidget(image_label, 0, Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(num)
        tile.setCursor(Qt.CursorShape.PointingHandCursor)
        tile.mousePressEvent = lambda event, i=index: self._toggle_excluded(i)
        return tile

    def _load_thumb_pixmap(self, path: str) -> QPixmap:
        try:
            with Image.open(path) as img:
                img.thumbnail((80, 80), Image.LANCZOS)
                return self._pil_to_pixmap(img.convert("RGB"))
        except Exception:
            return QPixmap()

    def _pil_to_pixmap(self, img: Image.Image) -> QPixmap:
        rgb = img.convert("RGB")
        data = rgb.tobytes("raw", "RGB")
        qimg = QImage(data, rgb.width, rgb.height, rgb.width * 3, QImage.Format.Format_RGB888).copy()
        return QPixmap.fromImage(qimg)

    def _scan_image_files(self, path: str) -> list[str]:
        exts = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}
        if not os.path.isdir(path):
            return []
        files = []
        for name in sorted(os.listdir(path)):
            full = os.path.join(path, name)
            if os.path.isfile(full) and os.path.splitext(name)[1].lower() in exts:
                files.append(full)
        return files

    def _step_card(self, num: str, title: str):
        card = QWidget()
        card.setObjectName("step_card")
        card.setStyleSheet(f"QWidget#step_card {{ background: {_CARD}; border: 1px solid {_SEP}; border-radius: 10px; }}")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(10)
        header = QHBoxLayout()
        badge = QLabel(num)
        badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        badge.setFixedSize(28, 28)
        badge.setStyleSheet(f"background: {_GREEN}; color: white; border-radius: 14px; font-weight: 700;")
        label = self._label(title, "h2")
        header.addWidget(badge)
        header.addWidget(label)
        header.addStretch(1)
        layout.addLayout(header)
        return card, layout

    # ── Behaviors ────────────────────────────────────────────────
    def _on_preset_clicked(self, name: str):
        rows, cols = (int(part) for part in name.split("×", 1))
        blockers = [QSignalBlocker(self._row_spin), QSignalBlocker(self._col_spin)]
        self._row_spin.setValue(rows)
        self._col_spin.setValue(cols)
        del blockers
        self._refresh_preset_state()
        self._refresh_preview()
        self._emit_config_changed()

    def _on_form_changed(self, *_args):
        self._current_collage = None
        self._refresh_preset_state()
        self._refresh_preview()
        self._emit_config_changed()

    def _on_background_changed(self, *_args):
        self._current_collage = None
        self._update_color_swatch()
        self._remember_background_color()
        self._emit_config_changed()

    def _on_template_item_clicked(self, item: QListWidgetItem):
        tpl = item.data(Qt.ItemDataRole.UserRole)
        if isinstance(tpl, CollageTemplate):
            self.set_config(tpl)

    def _save_template_to_disk(self, checked=False, name: str | None = None):
        del checked
        if name is None:
            name, ok = QInputDialog.getText(self, "保存模板", "模板名称：")
            if not ok:
                return
        name = name.strip()
        if not name:
            return
        if self._mgr.load(name) is not None:
            answer = QMessageBox.question(self, "覆盖模板", f"模板「{name}」已存在，是否覆盖？")
            if answer != QMessageBox.StandardButton.Yes:
                return

        tpl = self.get_current_config()
        tpl.name = name
        self._mgr.save(tpl)
        self._current_collage = tpl
        self._load_template_list(select_name=name)

    def _delete_selected_template(self):
        item = self._template_list.currentItem()
        if item is None:
            return
        name = item.text()
        if QMessageBox.question(self, "删除模板", f"确定删除「{name}」？") != QMessageBox.StandardButton.Yes:
            return
        self._mgr.delete(name)
        if self._current_collage and self._current_collage.name == name:
            self._current_collage = None
        self._load_template_list()

    def _choose_background_color(self):
        color = QColorDialog.getColor(QColor(self._background_edit.text()), self, "选择背景色")
        if color.isValid():
            self._background_edit.setText(color.name().upper())

    def _emit_config_changed(self):
        if not self._suppress_emit:
            self.config_changed.emit(self.get_current_config())

    # ── Helpers ──────────────────────────────────────────────────
    def _load_template_list(self, select_name: str | None = None):
        self._template_list.clear()
        for tpl in self._mgr.load_all():
            item = QListWidgetItem(tpl.name)
            item.setData(Qt.ItemDataRole.UserRole, tpl)
            self._template_list.addItem(item)
            if tpl.name == select_name:
                self._template_list.setCurrentItem(item)

    def _refresh_preset_state(self):
        current = f"{self._row_spin.value()}×{self._col_spin.value()}"
        for name, btn in self._preset_buttons.items():
            btn.setChecked(name == current)

    def _refresh_preview(self):
        while self._mini_grid.count():
            item = self._mini_grid.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        rows = self._row_spin.value()
        cols = self._col_spin.value()
        total = rows * cols
        self._preview_label.setText(f"布局预览  {rows}×{cols} = {total} 张/页")
        for index in range(total):
            cell = QLabel(str(index + 1))
            cell.setObjectName("mini_cell")
            cell.setAlignment(Qt.AlignmentFlag.AlignCenter)
            cell.setMinimumHeight(26)
            self._mini_grid.addWidget(cell, index // cols, index % cols)

    def _current_aspect_ratio(self) -> float:
        return _ASPECTS.get(self._aspect_combo.currentText(), 0)

    def _aspect_label_for(self, ratio: float) -> str:
        for label, value in _ASPECTS.items():
            if abs(value - ratio) < 0.001:
                return label
        return "自适应"

    def _last_background_color(self) -> str:
        return QSettings("融景", "RongJing").value(
            "collage/last_background_color", "#FFFFFF", type=str
        )

    def _remember_background_color(self):
        text = self._background_edit.text().strip()
        if QColor(text).isValid():
            QSettings("融景", "RongJing").setValue("collage/last_background_color", text.upper())

    def _update_color_swatch(self):
        color = self._background_edit.text().strip()
        if not QColor(color).isValid():
            color = "#FFFFFF"
        self._color_swatch.setStyleSheet(f"QLabel#color_swatch {{ background: {color}; }}")

    def _spin(self, minimum: int, maximum: int, value: int) -> QSpinBox:
        spin = QSpinBox()
        spin.setRange(minimum, maximum)
        spin.setValue(value)
        return spin

    def _add_grid_field(self, grid: QGridLayout, row: int, col: int, label: str, field):
        box = QVBoxLayout()
        box.setSpacing(4)
        box.addWidget(self._label(label, "cap"))
        if isinstance(field, QHBoxLayout):
            box.addLayout(field)
        else:
            box.addWidget(field)
        grid.addLayout(box, row, col)

    def _label(self, text: str, name: str | None = None) -> QLabel:
        label = QLabel(text)
        if name:
            label.setObjectName(name)
        return label

    def _sep(self) -> QFrame:
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setFrameShadow(QFrame.Shadow.Plain)
        return sep
