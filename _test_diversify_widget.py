import os
import tempfile

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtCore import QSettings
from PyQt6.QtWidgets import QApplication

from core.diversifier import DiversifyConfig
from ui.diversify_widget import DiversifyWidget


def _app():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def _settings():
    settings = QSettings("融景", "RongJing")
    settings.remove("diversify")
    settings.sync()
    return settings


def main():
    with tempfile.TemporaryDirectory() as tmpdir:
        QSettings.setDefaultFormat(QSettings.Format.IniFormat)
        QSettings.setPath(QSettings.Format.IniFormat, QSettings.Scope.UserScope, tmpdir)
        app = _app()
        assert app is not None

        settings = _settings()
        widget = DiversifyWidget()
        config = widget.get_config()
        assert isinstance(config, DiversifyConfig)
        assert config.enabled is False
        assert widget.SETTINGS_PREFIX == "diversify"

        emissions = []
        widget.config_changed.connect(emissions.append)
        widget._enabled_checkbox.setChecked(True)
        assert len(emissions) == 1
        assert emissions[-1].enabled is True
        assert settings.value("diversify/enabled", False, type=bool) is True

        widget._low_radio.setChecked(True)
        low = widget.get_config()
        assert low.enabled is True
        assert low.corner_jitter_px == DiversifyConfig.preset("low").corner_jitter_px
        assert settings.value("diversify/strength", "", type=str) == "low"

        widget._medium_radio.setChecked(True)
        medium = widget.get_config()
        assert medium.corner_jitter_px == DiversifyConfig.preset("medium").corner_jitter_px
        assert settings.value("diversify/strength", "", type=str) == "medium"

        restored = DiversifyWidget()
        assert restored.get_config().enabled is True
        assert restored._medium_radio.isChecked()

        before_enabled = settings.value("diversify/enabled", False, type=bool)
        before_strength = settings.value("diversify/strength", "", type=str)
        set_emissions = []
        restored.config_changed.connect(set_emissions.append)
        restored.set_config(DiversifyConfig.preset("high"))
        assert restored.get_config().enabled is True
        assert restored._high_radio.isChecked()
        assert set_emissions == []
        assert settings.value("diversify/enabled", False, type=bool) == before_enabled
        assert settings.value("diversify/strength", "", type=str) == before_strength

        restored._reset_button.click()
        assert restored.get_config().enabled is False
        assert restored._medium_radio.isChecked()
        assert settings.value("diversify/enabled", True, type=bool) is False
        assert settings.value("diversify/strength", "", type=str) == "medium"


if __name__ == "__main__":
    main()
