from PyQt6.QtCore import QSettings, QSignalBlocker, pyqtSignal
from PyQt6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QRadioButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from core.diversifier import DiversifyConfig


_GREEN = "#07C160"
_CARD = "#FFFFFF"
_INPUT = "#F0F0F0"
_SEP = "#E5E5E5"
_TEXT = "#191919"
_TEXT2 = "#888888"


class DiversifyWidget(QWidget):
    config_changed = pyqtSignal(object)

    SETTINGS_PREFIX = "diversify"

    def __init__(self, parent=None):
        super().__init__(parent)
        self._build_ui()
        self._load_from_settings()
        self._wire_signals()

    def get_config(self) -> DiversifyConfig:
        config = DiversifyConfig.preset(self._current_strength())
        config.enabled = self._enabled_checkbox.isChecked()
        return config

    def set_config(self, config: DiversifyConfig):
        blockers = [
            QSignalBlocker(self._enabled_checkbox),
            QSignalBlocker(self._low_radio),
            QSignalBlocker(self._medium_radio),
            QSignalBlocker(self._high_radio),
            QSignalBlocker(self._strength_group),
        ]
        self._enabled_checkbox.setChecked(config.enabled)
        self._radio_for_strength(self._strength_from_config(config)).setChecked(True)
        del blockers

    def _build_ui(self):
        self.setObjectName("DiversifyWidget")
        self.setStyleSheet(
            f"""
            QWidget#DiversifyWidget {{
                background: {_CARD};
                border: 1px solid {_SEP};
                border-radius: 10px;
            }}
            QWidget#DiversifyWidget QLabel {{
                background: transparent;
                color: {_TEXT};
                border: none;
            }}
            QWidget#DiversifyWidget QLabel#hint {{
                color: {_TEXT2};
                font-size: 12px;
            }}
            QWidget#DiversifyWidget QCheckBox,
            QWidget#DiversifyWidget QRadioButton {{
                background: transparent;
                color: {_TEXT};
                spacing: 8px;
                border: none;
            }}
            QWidget#DiversifyWidget QCheckBox::indicator,
            QWidget#DiversifyWidget QRadioButton::indicator {{
                width: 16px;
                height: 16px;
                border: 2px solid {_SEP};
                background: {_CARD};
            }}
            QWidget#DiversifyWidget QCheckBox::indicator {{
                border-radius: 4px;
            }}
            QWidget#DiversifyWidget QRadioButton::indicator {{
                border-radius: 9px;
            }}
            QWidget#DiversifyWidget QCheckBox::indicator:checked,
            QWidget#DiversifyWidget QRadioButton::indicator:checked {{
                background: {_GREEN};
                border-color: {_GREEN};
            }}
            QWidget#DiversifyWidget QPushButton {{
                background: {_INPUT};
                color: {_TEXT};
                border: 1px solid {_SEP};
                border-radius: 8px;
                padding: 7px 14px;
                font-size: 13px;
            }}
            QWidget#DiversifyWidget QPushButton:hover {{
                background: #E5E5E5;
            }}
            """
        )

        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(10)

        self._enabled_checkbox = QCheckBox("启用批次差异化")
        layout.addWidget(self._enabled_checkbox)

        strength_row = QHBoxLayout()
        strength_row.setSpacing(14)
        self._low_radio = QRadioButton("低")
        self._medium_radio = QRadioButton("中（推荐）")
        self._high_radio = QRadioButton("高")
        strength_row.addWidget(self._low_radio)
        strength_row.addWidget(self._medium_radio)
        strength_row.addWidget(self._high_radio)
        strength_row.addStretch(1)
        layout.addLayout(strength_row)

        self._strength_group = QButtonGroup(self)
        self._strength_group.setExclusive(True)
        self._strength_group.addButton(self._low_radio)
        self._strength_group.addButton(self._medium_radio)
        self._strength_group.addButton(self._high_radio)

        self._hint_label = QLabel("让每张输出有可见但克制的差异，避免视觉雷同")
        self._hint_label.setObjectName("hint")
        self._hint_label.setWordWrap(True)
        layout.addWidget(self._hint_label)

        self._reset_button = QPushButton("恢复默认")
        self._reset_button.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        layout.addWidget(self._reset_button)

    def _load_from_settings(self):
        settings = QSettings("融景", "RongJing")
        enabled = settings.value(f"{self.SETTINGS_PREFIX}/enabled", False, type=bool)
        strength = settings.value(f"{self.SETTINGS_PREFIX}/strength", "medium", type=str)
        if strength not in {"low", "medium", "high"}:
            strength = "medium"

        self._enabled_checkbox.setChecked(bool(enabled))
        self._radio_for_strength(strength).setChecked(True)

    def _save_to_settings(self):
        settings = QSettings("融景", "RongJing")
        settings.setValue(f"{self.SETTINGS_PREFIX}/enabled", self._enabled_checkbox.isChecked())
        settings.setValue(f"{self.SETTINGS_PREFIX}/strength", self._current_strength())
        settings.sync()

    def _wire_signals(self):
        self._enabled_checkbox.toggled.connect(self._on_control_changed)
        self._low_radio.toggled.connect(lambda checked: self._on_strength_toggled(checked))
        self._medium_radio.toggled.connect(lambda checked: self._on_strength_toggled(checked))
        self._high_radio.toggled.connect(lambda checked: self._on_strength_toggled(checked))
        self._reset_button.clicked.connect(self._reset_to_defaults)

    def _on_strength_toggled(self, checked: bool):
        if checked:
            self._on_control_changed()

    def _on_control_changed(self):
        self._save_to_settings()
        self.config_changed.emit(self.get_config())

    def _reset_to_defaults(self):
        blockers = [
            QSignalBlocker(self._enabled_checkbox),
            QSignalBlocker(self._low_radio),
            QSignalBlocker(self._medium_radio),
            QSignalBlocker(self._high_radio),
            QSignalBlocker(self._strength_group),
        ]
        self._enabled_checkbox.setChecked(False)
        self._medium_radio.setChecked(True)
        del blockers
        self._on_control_changed()

    def _current_strength(self) -> str:
        if self._low_radio.isChecked():
            return "low"
        if self._high_radio.isChecked():
            return "high"
        return "medium"

    def _radio_for_strength(self, strength: str) -> QRadioButton:
        if strength == "low":
            return self._low_radio
        if strength == "high":
            return self._high_radio
        return self._medium_radio

    def _strength_from_config(self, config: DiversifyConfig) -> str:
        compared_fields = (
            "corner_jitter_px",
            "brightness_range",
            "contrast_range",
            "saturation_range",
            "rotation_range",
            "scale_range",
            "noise_intensity",
            "jpeg_quality_range",
            "randomize_metadata",
        )
        for strength in ("low", "medium", "high"):
            preset = DiversifyConfig.preset(strength)
            if all(getattr(config, field) == getattr(preset, field) for field in compared_fields):
                return strength
        return "medium"
