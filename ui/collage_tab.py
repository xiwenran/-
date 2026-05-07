import math
import os
import subprocess
from pathlib import Path
from typing import Optional

from PIL import Image
from PyQt6.QtCore import QSettings, QSignalBlocker, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QColor, QDragEnterEvent, QDropEvent, QImage, QPixmap
from PyQt6.QtWidgets import (
    QColorDialog,
    QComboBox,
    QFileDialog,
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
from core.diversifier import DiversifyConfig, diversify_image
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

_PRESETS = ("2×1", "3×1", "4×1", "2×2", "3×2", "4×2")
_ASPECTS = {
    "16:9": 16 / 9,
    "4:3": 4 / 3,
    "3:4": 3 / 4,
    "1:1": 1,
    "自适应": 0,
}

_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}


class CollageTab(QWidget):
    """拼图 Tab — 固定高度两栏布局，预览区占右侧主体。"""

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
        self._preview_collage_index = 0
        self._collage_runner: CollageBatchRunner | None = None
        self._cached_preview: Image.Image | None = None
        self._compare_mode = False
        self._batch_mode = False
        self._subfolder_items: list[tuple[str, list[str]]] = []

        self._app_data_dir = str(Path(collages_dir).parent)

        self.setAcceptDrops(True)
        self._build_ui()
        self._load_template_list()
        self._wire_signals()
        self._refresh_preset_state()
        self._refresh_mini_preview()
        self._restore_last_dirs()

    # ── Public API ────────────────────────────────────────────────
    def get_current_config(self) -> CollageTemplate:
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
        self._refresh_mini_preview()
        del blockers
        self._suppress_emit = False
        self._refresh_state()

    def get_diversify_config(self):
        return self._diversify.get_config()

    # ── Drag & drop ───────────────────────────────────────────────
    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent):
        urls = event.mimeData().urls()
        if not urls:
            return
        path = urls[0].toLocalFile()
        if path.lower().endswith((".pptx", ".ppt")):
            self._import_pptx(path)
        elif os.path.isdir(path):
            self._set_input_dir(path)
        elif os.path.isfile(path) and os.path.splitext(path)[1].lower() in _IMAGE_EXTS:
            self._set_input_dir(os.path.dirname(path))

    # ── UI build ──────────────────────────────────────────────────
    def _build_ui(self):
        self.setObjectName("CollageTab")
        self.setStyleSheet(self._stylesheet())

        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        root.addWidget(self._build_sidebar(), 0)
        root.addWidget(self._build_right_panel(), 1)

    def _build_sidebar(self) -> QWidget:
        sidebar = QWidget()
        sidebar.setObjectName("collage_sidebar")
        sidebar.setFixedWidth(340)
        outer = QVBoxLayout(sidebar)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        body = QWidget()
        body.setObjectName("collage_scroll_body")
        content = QVBoxLayout(body)
        content.setContentsMargins(16, 18, 16, 18)
        content.setSpacing(10)

        self._add_input_section(content)
        content.addWidget(self._sep())
        self._add_layout_section(content)
        content.addWidget(self._sep())
        self._add_split_section(content)
        content.addWidget(self._sep())
        self._add_template_section(content)
        content.addWidget(self._sep())
        content.addWidget(self._label("批次差异化", "h2"))
        self._diversify = DiversifyWidget(self)
        content.addWidget(self._diversify)
        content.addStretch(1)

        scroll.setWidget(body)
        outer.addWidget(scroll)
        return sidebar

    def _build_right_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(10)

        layout.addWidget(self._build_preview_area(), 1)
        layout.addWidget(self._build_thumbnail_area(), 0)
        layout.addWidget(self._sep())
        layout.addWidget(self._build_export_bar(), 0)
        return panel

    # ── Sidebar sections ──────────────────────────────────────────
    def _add_input_section(self, content: QVBoxLayout):
        content.addWidget(self._label("输入来源", "h2"))

        mode_row = QHBoxLayout()
        mode_row.setSpacing(8)
        self._single_mode_btn = QPushButton("单个文件夹")
        self._single_mode_btn.setCheckable(True)
        self._single_mode_btn.setChecked(True)
        self._single_mode_btn.setObjectName("preset_btn")
        self._batch_mode_btn = QPushButton("批量文件夹")
        self._batch_mode_btn.setCheckable(True)
        self._batch_mode_btn.setObjectName("preset_btn")
        mode_row.addWidget(self._single_mode_btn)
        mode_row.addWidget(self._batch_mode_btn)
        content.addLayout(mode_row)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(8)
        self._choose_dir_btn = QPushButton("选择文件夹")
        self._choose_dir_btn.clicked.connect(self._choose_input_dir)
        self._import_ppt_btn = QPushButton("导入 PPT")
        self._import_ppt_btn.clicked.connect(self._choose_pptx)
        btn_row.addWidget(self._choose_dir_btn)
        btn_row.addWidget(self._import_ppt_btn)
        content.addLayout(btn_row)

        self._input_path_label = self._label("未选择（也可直接拖入文件夹或 PPT）", "hint")
        self._input_path_label.setWordWrap(True)
        content.addWidget(self._input_path_label)
        self._input_count_label = self._label("", "hint")
        content.addWidget(self._input_count_label)

        self._subfolder_list = QListWidget()
        self._subfolder_list.setMaximumHeight(140)
        self._subfolder_list.setVisible(False)
        content.addWidget(self._subfolder_list)

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

        self._mini_preview_label = self._label("", "cap")
        content.addWidget(self._mini_preview_label)
        self._mini_grid_frame = QWidget()
        self._mini_grid_frame.setObjectName("mini_grid_frame")
        self._mini_grid = QGridLayout(self._mini_grid_frame)
        self._mini_grid.setContentsMargins(8, 8, 8, 8)
        self._mini_grid.setSpacing(5)
        content.addWidget(self._mini_grid_frame)

    def _add_split_section(self, content: QVBoxLayout):
        content.addWidget(self._label("自动拆分", "h2"))
        row = QHBoxLayout()
        row.setSpacing(8)
        row.addWidget(self._label("输出", "cap"))
        self._output_count_spin = self._spin(1, 1, 1)
        self._output_count_spin.setFixedWidth(70)
        row.addWidget(self._output_count_spin)
        self._pages_per_label = self._label("张，每张 0 页", "cap")
        row.addWidget(self._pages_per_label)
        row.addStretch(1)
        content.addLayout(row)

        self._selected_pages_label = self._label("已选 0/0 页", "hint")
        content.addWidget(self._selected_pages_label)

    def _add_template_section(self, content: QVBoxLayout):
        content.addWidget(self._label("拼图模板", "h2"))
        tpl_frame = QWidget()
        tpl_frame.setObjectName("tpl_list_frame")
        frame_layout = QVBoxLayout(tpl_frame)
        frame_layout.setContentsMargins(4, 4, 4, 4)
        self._template_list = QListWidget()
        self._template_list.setMinimumHeight(80)
        self._template_list.setMaximumHeight(150)
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

    # ── Right panel sections ──────────────────────────────────────
    def _build_preview_area(self) -> QWidget:
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        self._collage_preview_label = QLabel("选择文件夹或拖入 PPT 开始预览")
        self._collage_preview_label.setObjectName("hint")
        self._collage_preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._collage_preview_label.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self._collage_preview_label.setStyleSheet(
            f"background: {_CARD}; border: 1px solid {_SEP}; border-radius: 8px;"
        )
        layout.addWidget(self._collage_preview_label, 1)

        nav_row = QHBoxLayout()
        nav_row.setSpacing(8)
        self._preview_prev_btn = QPushButton("◀ 上一张")
        self._preview_next_btn = QPushButton("下一张 ▶")
        self._preview_prev_btn.clicked.connect(lambda: self._change_preview(-1))
        self._preview_next_btn.clicked.connect(lambda: self._change_preview(1))
        self._preview_info_label = self._label("", "hint")
        self._preview_info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self._compare_btn = QPushButton("对比差异化")
        self._compare_btn.setCheckable(True)
        self._compare_btn.setVisible(False)
        self._compare_btn.clicked.connect(self._toggle_compare)

        nav_row.addWidget(self._preview_prev_btn)
        nav_row.addWidget(self._preview_info_label, 1)
        nav_row.addWidget(self._compare_btn)
        nav_row.addWidget(self._preview_next_btn)
        layout.addLayout(nav_row)

        return container

    def _build_thumbnail_area(self) -> QWidget:
        container = QWidget()
        container.setFixedHeight(170)
        outer = QVBoxLayout(container)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(4)

        header = QHBoxLayout()
        header.addWidget(self._label("页面缩略图（点击排除/恢复）", "cap"))
        header.addStretch(1)
        outer.addLayout(header)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)

        self._thumb_body = QWidget()
        self._thumb_grid = QHBoxLayout(self._thumb_body)
        self._thumb_grid.setContentsMargins(0, 0, 0, 0)
        self._thumb_grid.setSpacing(6)
        scroll.setWidget(self._thumb_body)
        outer.addWidget(scroll, 1)

        return container

    def _build_export_bar(self) -> QWidget:
        bar = QWidget()
        layout = QHBoxLayout(bar)
        layout.setContentsMargins(0, 4, 0, 4)
        layout.setSpacing(12)

        self._choose_output_btn = QPushButton("输出文件夹")
        self._choose_output_btn.clicked.connect(self._choose_output_dir)
        self._output_path_label = self._label("未选择", "hint")
        self._output_path_label.setMinimumWidth(60)
        self._format_combo = QComboBox()
        self._format_combo.addItems(["PNG", "JPEG"])
        self._run_collage_btn = QPushButton("开始拼图")
        self._run_collage_btn.setObjectName("primary")
        self._run_collage_btn.setMinimumHeight(40)
        self._run_collage_btn.clicked.connect(self._run_collage_batch)
        self._cancel_collage_btn = QPushButton("取消")
        self._cancel_collage_btn.setVisible(False)
        self._cancel_collage_btn.clicked.connect(self._abort_collage_batch)

        layout.addWidget(self._choose_output_btn, 0)
        layout.addWidget(self._output_path_label, 1)
        layout.addWidget(self._format_combo, 0)
        layout.addWidget(self._run_collage_btn, 0)
        layout.addWidget(self._cancel_collage_btn, 0)

        self._collage_progress = QProgressBar()
        self._collage_progress.setVisible(False)
        self._collage_status = self._label("", "hint")
        self._collage_status.setVisible(False)

        wrapper = QWidget()
        wl = QVBoxLayout(wrapper)
        wl.setContentsMargins(0, 0, 0, 0)
        wl.setSpacing(4)
        wl.addWidget(bar)
        wl.addWidget(self._collage_progress)
        wl.addWidget(self._collage_status)
        return wrapper

    # ── Signal wiring ─────────────────────────────────────────────
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
        self._diversify.config_changed.connect(self._on_diversify_changed)
        self.config_changed.connect(lambda _cfg: self._refresh_state())
        self._output_count_spin.valueChanged.connect(self._on_output_count_changed)
        self._single_mode_btn.clicked.connect(lambda: self._set_batch_mode(False))
        self._batch_mode_btn.clicked.connect(lambda: self._set_batch_mode(True))
        self._subfolder_list.currentRowChanged.connect(self._on_subfolder_selected)

    # ── Input handling ────────────────────────────────────────────
    def _choose_input_dir(self):
        start = self._input_dir or ""
        path = QFileDialog.getExistingDirectory(self, "选择图片文件夹", start)
        if path:
            self._set_input_dir(path)

    def _choose_pptx(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "选择 PPT 文件", "", "PowerPoint (*.pptx *.ppt)"
        )
        if path:
            self._import_pptx(path)

    def _import_pptx(self, pptx_path: str):
        export_dir = os.path.join(self._app_data_dir, "ppt_export", Path(pptx_path).stem)
        os.makedirs(export_dir, exist_ok=True)

        script = (
            'tell application "Microsoft PowerPoint"\n'
            f'    open POSIX file "{pptx_path}"\n'
            '    set pres to active presentation\n'
            f'    save pres in POSIX file "{export_dir}" as save as PNG\n'
            '    close pres saving no\n'
            'end tell'
        )
        self._input_path_label.setText(f"正在导出 PPT…")
        from PyQt6.QtWidgets import QApplication
        QApplication.processEvents()

        result = subprocess.run(
            ["osascript", "-e", script],
            capture_output=True, text=True, timeout=120,
        )
        if result.returncode != 0:
            QMessageBox.warning(
                self, "PPT 导出失败",
                "无法调用 PowerPoint 导出图片。\n"
                "请确认已安装 Microsoft PowerPoint for Mac。\n\n"
                f"错误信息：{result.stderr[:200]}"
            )
            self._input_path_label.setText("PPT 导出失败")
            return

        self._set_input_dir(export_dir)

    def _set_input_dir(self, path: str):
        self._input_dir = path
        QSettings("融景", "RongJing").setValue("collage/last_input_dir", path)

        if self._batch_mode:
            self._scan_subfolders(path)
        else:
            self._image_files = self._scan_image_files(path)
            self._excluded_indices = set()
            self._preview_collage_index = 0
            self._input_path_label.setText(path)
            self._input_count_label.setText(f"已扫描到 {len(self._image_files)} 张图片")
            self._reset_output_count()
            self._refresh_thumbnails()
            self._refresh_state()

    def _set_batch_mode(self, batch: bool):
        self._batch_mode = batch
        self._single_mode_btn.setChecked(not batch)
        self._batch_mode_btn.setChecked(batch)
        self._subfolder_list.setVisible(batch)
        if batch and self._input_dir:
            self._scan_subfolders(self._input_dir)
        elif not batch and self._input_dir:
            self._set_input_dir(self._input_dir)

    def _scan_subfolders(self, path: str):
        self._subfolder_items = []
        self._subfolder_list.clear()
        if not os.path.isdir(path):
            return
        for name in sorted(os.listdir(path)):
            sub = os.path.join(path, name)
            if os.path.isdir(sub):
                files = self._scan_image_files(sub)
                if files:
                    self._subfolder_items.append((name, files))
                    self._subfolder_list.addItem(f"{name}  ({len(files)} 张)")

        total_files = sum(len(f) for _, f in self._subfolder_items)
        self._input_path_label.setText(path)
        self._input_count_label.setText(
            f"共 {len(self._subfolder_items)} 个子文件夹，{total_files} 张图片"
        )

        if self._subfolder_items:
            self._subfolder_list.setCurrentRow(0)
            self._on_subfolder_selected(0)

    def _on_subfolder_selected(self, row: int):
        if row < 0 or row >= len(self._subfolder_items):
            return
        _, files = self._subfolder_items[row]
        self._image_files = files
        self._excluded_indices = set()
        self._preview_collage_index = 0
        self._reset_output_count()
        self._refresh_thumbnails()
        self._refresh_state()

    def _choose_output_dir(self):
        path = QFileDialog.getExistingDirectory(self, "选择输出文件夹", self._output_dir)
        if path:
            self._output_dir = path
            QSettings("融景", "RongJing").setValue("collage/last_output_dir", path)
            self._output_path_label.setText(path)

    def _restore_last_dirs(self):
        s = QSettings("融景", "RongJing")
        last_in = s.value("collage/last_input_dir", "", type=str)
        if last_in:
            self._input_path_label.setText(last_in)
        last_out = s.value("collage/last_output_dir", "", type=str)
        if last_out:
            self._output_dir = last_out
            self._output_path_label.setText(last_out)

    # ── Scan files (recursive) ────────────────────────────────────
    def _scan_image_files(self, path: str) -> list[str]:
        if not os.path.isdir(path):
            return []
        files = []
        for root, _dirs, names in os.walk(path):
            for name in sorted(names):
                if os.path.splitext(name)[1].lower() in _IMAGE_EXTS:
                    files.append(os.path.join(root, name))
        files.sort(key=lambda p: self._natural_key(p))
        return files

    @staticmethod
    def _natural_key(s: str):
        import re
        return [int(c) if c.isdigit() else c.lower() for c in re.split(r'(\d+)', s)]

    # ── Preview ───────────────────────────────────────────────────
    def _refresh_collage_preview(self):
        ranges = self._current_ranges()
        if not ranges:
            self._cached_preview = None
            self._collage_preview_label.setPixmap(QPixmap())
            self._collage_preview_label.setText("选择文件夹或拖入 PPT 开始预览")
            self._preview_info_label.setText("")
            return

        idx = min(self._preview_collage_index, len(ranges) - 1)
        self._preview_collage_index = idx
        selected = self._selected_image_files()
        start, end = ranges[idx]
        images = []
        try:
            for p in selected[start:end]:
                with Image.open(p) as img:
                    images.append(img.copy())
            cfg = self.get_current_config()
            preview = create_collage(
                images, cfg.layout, cfg.rows, cfg.cols,
                gap=cfg.gap, padding=cfg.padding,
                background_color=cfg.background_color,
                cell_aspect_ratio=cfg.cell_aspect_ratio,
                output_width=1200, output_height=0,
            )
            self._cached_preview = preview
            self._display_preview(preview)
            self._preview_info_label.setText(
                f"第 {idx + 1}/{len(ranges)} 张 · {end - start} 页"
            )
        except Exception as exc:
            self._cached_preview = None
            self._collage_preview_label.setPixmap(QPixmap())
            self._collage_preview_label.setText(f"预览失败：{exc}")

    def _display_preview(self, img: Image.Image):
        if self._compare_mode:
            cfg = self._diversify.get_config()
            if cfg.enabled:
                varied = diversify_image(img, cfg, seed=42)
                w = img.width
                combined = Image.new("RGB", (w * 2 + 4, img.height), (200, 200, 200))
                combined.paste(img, (0, 0))
                combined.paste(varied, (w + 4, 0))
                img = combined

        label = self._collage_preview_label
        lw, lh = label.width() - 4, label.height() - 4
        if lw < 100 or lh < 100:
            lw, lh = 600, 400
        display = img.copy()
        display.thumbnail((lw, lh), Image.LANCZOS)
        label.setText("")
        label.setPixmap(self._pil_to_pixmap(display))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self._cached_preview:
            QTimer.singleShot(50, lambda: self._display_preview(self._cached_preview))

    def _change_preview(self, delta: int):
        ranges = self._current_ranges()
        if not ranges:
            return
        new_idx = self._preview_collage_index + delta
        if 0 <= new_idx < len(ranges):
            self._preview_collage_index = new_idx
            self._refresh_collage_preview()

    def _toggle_compare(self):
        self._compare_mode = self._compare_btn.isChecked()
        if self._cached_preview:
            self._display_preview(self._cached_preview)

    # ── Thumbnails ────────────────────────────────────────────────
    def _refresh_thumbnails(self):
        while self._thumb_grid.count():
            item = self._thumb_grid.takeAt(0)
            w = item.widget()
            if w:
                w.deleteLater()

        for index, path in enumerate(self._image_files):
            self._thumb_grid.addWidget(self._thumbnail_widget(index, path))
        self._thumb_grid.addStretch(1)

    def _thumbnail_widget(self, index: int, path: str) -> QWidget:
        tile = QWidget()
        tile.setObjectName("thumb_tile")
        tile.setFixedSize(80, 100)
        excluded = index in self._excluded_indices
        border = _RED if excluded else _GREEN
        tile.setStyleSheet(
            f"QWidget#thumb_tile {{ background: white; border: 2px solid {border}; border-radius: 6px; }}"
        )
        layout = QVBoxLayout(tile)
        layout.setContentsMargins(3, 3, 3, 3)
        layout.setSpacing(2)

        image_label = QLabel()
        image_label.setFixedSize(72, 72)
        image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        image_label.setPixmap(self._load_thumb_pixmap(path))

        mark = QLabel("×" if excluded else "✓", tile)
        mark.setAlignment(Qt.AlignmentFlag.AlignCenter)
        mark.setStyleSheet(
            f"background: {border}; color: white; border-radius: 8px; font-weight: 700;"
        )
        mark.setFixedSize(16, 16)
        mark.move(58, 3)

        num = QLabel(f"P{index + 1}")
        num.setAlignment(Qt.AlignmentFlag.AlignCenter)
        num.setStyleSheet("border: 0; color: #666; font-size: 10px;")
        layout.addWidget(image_label, 0, Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(num)
        tile.setCursor(Qt.CursorShape.PointingHandCursor)
        tile.mousePressEvent = lambda event, i=index: self._toggle_excluded(i)
        return tile

    def _toggle_excluded(self, index: int):
        if index < 0 or index >= len(self._image_files):
            return
        if index in self._excluded_indices:
            self._excluded_indices.remove(index)
        else:
            self._excluded_indices.add(index)
        self._refresh_thumbnails()
        self._refresh_state()

    # ── Batch run ─────────────────────────────────────────────────
    def _run_collage_batch(self, callback=None):
        if not self._output_dir:
            QMessageBox.warning(self, "提示", "请先选择输出文件夹")
            return

        if self._batch_mode:
            self._run_batch_multi_folder(callback)
        else:
            self._run_batch_single(callback)

    def _run_batch_single(self, callback=None):
        if not self._image_files:
            QMessageBox.warning(self, "提示", "请先选择图片文件夹")
            return
        cfg = self.get_current_config()
        diversify_cfg = self._diversify.get_config()
        self._collage_runner = CollageBatchRunner(
            self._image_files, cfg, self._output_dir,
            self._format_combo.currentText(),
            self._output_count_spin.value(),
            set(self._excluded_indices),
            diversify_cfg, self,
        )
        self._start_runner(callback)

    def _run_batch_multi_folder(self, callback=None):
        if not self._subfolder_items:
            QMessageBox.warning(self, "提示", "未找到包含图片的子文件夹")
            return

        cfg = self.get_current_config()
        diversify_cfg = self._diversify.get_config()
        total_cells = max(1, cfg.total_cells)

        self._batch_queue = list(self._subfolder_items)
        self._batch_done = 0
        self._batch_total = len(self._batch_queue)
        self._batch_callback = callback
        self._batch_cfg = cfg
        self._batch_diversify_cfg = diversify_cfg
        self._run_next_in_queue()

    def _run_next_in_queue(self):
        if not self._batch_queue:
            self._on_collage_finished(True, f"全部完成！共处理 {self._batch_total} 个文件夹")
            if self._batch_callback:
                self._batch_callback(True, "")
            return

        name, files = self._batch_queue.pop(0)
        out_dir = os.path.join(self._output_dir, name)
        cfg = self._batch_cfg
        total_cells = max(1, cfg.total_cells)
        output_count = max(1, math.ceil(len(files) / total_cells))

        self._collage_runner = CollageBatchRunner(
            files, cfg, out_dir,
            self._format_combo.currentText(),
            output_count, set(),
            self._batch_diversify_cfg, self,
        )
        self._collage_runner.progress.connect(self._on_collage_progress)
        self._collage_runner.finished.connect(self._on_batch_item_finished)
        self._show_running_ui()
        self._collage_status.setText(f"[{self._batch_done + 1}/{self._batch_total}] {name}")
        self._collage_runner.start()

    def _on_batch_item_finished(self, success: bool, msg: str):
        self._batch_done += 1
        if not success:
            self._on_collage_finished(False, msg)
            return
        self._run_next_in_queue()

    def _start_runner(self, callback=None):
        self._collage_runner.progress.connect(self._on_collage_progress)
        self._collage_runner.finished.connect(self._on_collage_finished)
        if callback:
            self._collage_runner.finished.connect(callback)
        self._show_running_ui()
        self._collage_runner.start()

    def _show_running_ui(self):
        self._run_collage_btn.setVisible(False)
        self._cancel_collage_btn.setVisible(True)
        self._collage_progress.setVisible(True)
        self._collage_status.setVisible(True)
        self._collage_progress.setValue(0)
        self._collage_status.setText("正在拼图…")

    def _abort_collage_batch(self):
        if self._collage_runner:
            self._collage_runner.abort()
        self._batch_queue = []

    def _on_collage_progress(self, done: int, total: int, msg: str):
        self._collage_progress.setMaximum(max(1, total))
        self._collage_progress.setValue(done)
        self._collage_status.setText(f"[{done}/{total}] {msg}")

    def _on_collage_finished(self, success: bool, msg: str):
        self._run_collage_btn.setVisible(True)
        self._cancel_collage_btn.setVisible(False)
        self._collage_status.setText(msg)
        if os.environ.get("QT_QPA_PLATFORM") != "offscreen":
            if success:
                QMessageBox.information(self, "完成", msg)
            else:
                QMessageBox.warning(self, "处理结果", msg)

    # ── State refresh ─────────────────────────────────────────────
    def _refresh_state(self):
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
        self._pages_per_label.setText(f"张，每张 {pages_per} 页")
        self._run_collage_btn.setText(f"开始拼图（{len(ranges)} 张）")
        self._preview_prev_btn.setEnabled(self._preview_collage_index > 0)
        self._preview_next_btn.setEnabled(self._preview_collage_index < len(ranges) - 1)

        cfg = self._diversify.get_config()
        self._compare_btn.setVisible(cfg.enabled and self._cached_preview is not None)

        self._refresh_collage_preview()

    def _reset_output_count(self):
        selected_count = len(self._selected_image_files())
        cells = max(1, self.get_current_config().total_cells)
        value = max(1, math.ceil(selected_count / cells)) if selected_count else 1
        self._output_count_spin.setMaximum(max(1, selected_count))
        self._output_count_spin.setValue(value)

    def _selected_image_files(self) -> list[str]:
        return [p for i, p in enumerate(self._image_files) if i not in self._excluded_indices]

    def _current_ranges(self):
        return calculate_auto_split(
            len(self._selected_image_files()),
            self._output_count_spin.value(),
            max(1, self.get_current_config().total_cells),
        )

    # ── Behaviors ─────────────────────────────────────────────────
    def _on_preset_clicked(self, name: str):
        rows, cols = (int(part) for part in name.split("×", 1))
        blockers = [QSignalBlocker(self._row_spin), QSignalBlocker(self._col_spin)]
        self._row_spin.setValue(rows)
        self._col_spin.setValue(cols)
        del blockers
        self._refresh_preset_state()
        self._refresh_mini_preview()
        self._emit_config_changed()

    def _on_form_changed(self, *_args):
        self._current_collage = None
        self._refresh_preset_state()
        self._refresh_mini_preview()
        self._emit_config_changed()

    def _on_background_changed(self, *_args):
        self._current_collage = None
        self._update_color_swatch()
        self._remember_background_color()
        self._emit_config_changed()

    def _on_output_count_changed(self, *_args):
        self._refresh_state()

    def _on_template_item_clicked(self, item: QListWidgetItem):
        tpl = item.data(Qt.ItemDataRole.UserRole)
        if isinstance(tpl, CollageTemplate):
            self.set_config(tpl)

    def _on_diversify_changed(self, _cfg):
        cfg = self._diversify.get_config()
        self._compare_btn.setVisible(cfg.enabled and self._cached_preview is not None)
        if self._compare_mode and not cfg.enabled:
            self._compare_mode = False
            self._compare_btn.setChecked(False)
        if self._cached_preview:
            self._display_preview(self._cached_preview)
        self._emit_config_changed()

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

    # ── Helpers ───────────────────────────────────────────────────
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

    def _refresh_mini_preview(self):
        while self._mini_grid.count():
            item = self._mini_grid.takeAt(0)
            w = item.widget()
            if w:
                w.deleteLater()
        rows = self._row_spin.value()
        cols = self._col_spin.value()
        total = rows * cols
        self._mini_preview_label.setText(f"布局预览  {rows}×{cols} = {total} 张/页")
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

    def _load_thumb_pixmap(self, path: str) -> QPixmap:
        try:
            with Image.open(path) as img:
                img.thumbnail((72, 72), Image.LANCZOS)
                return self._pil_to_pixmap(img.convert("RGB"))
        except Exception:
            return QPixmap()

    def _pil_to_pixmap(self, img: Image.Image) -> QPixmap:
        rgb = img.convert("RGB")
        data = rgb.tobytes("raw", "RGB")
        qimg = QImage(data, rgb.width, rgb.height, rgb.width * 3, QImage.Format.Format_RGB888).copy()
        return QPixmap.fromImage(qimg)

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

    def _stylesheet(self) -> str:
        return f"""
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
