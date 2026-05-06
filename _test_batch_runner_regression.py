import hashlib
import os
import shutil
import tempfile

from PIL import Image
from PyQt6.QtCore import QCoreApplication, QEventLoop, QTimer

from core.batch_runner import BatchRunner
from core.diversifier import DiversifyConfig
from models.template_model import Template


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def make_fixture(root: str):
    bg_path = os.path.join(root, "background.png")
    ppt_path = os.path.join(root, "ppt.png")
    Image.new("RGB", (64, 64), (20, 40, 60)).save(bg_path)
    Image.new("RGB", (40, 40), (180, 120, 40)).save(ppt_path)
    template = Template(
        name="测试模板",
        background_path=bg_path,
        screen_points=[[0, 0], [64, 0], [64, 64], [0, 64]],
    )
    return template, ppt_path


def run_batch(tasks, output_dir: str, diversify_config=None) -> str:
    app = QCoreApplication.instance() or QCoreApplication([])
    loop = QEventLoop()
    result = {}
    runner = BatchRunner(
        tasks=tasks,
        output_dir=output_dir,
        output_format="PNG",
        diversify_config=diversify_config,
    )
    runner.finished.connect(lambda success, msg: (result.update(success=success, msg=msg), loop.quit()))
    QTimer.singleShot(15000, lambda: (result.update(success=False, msg="timeout"), runner.abort(), loop.quit()))
    runner.start()
    loop.exec()
    runner.wait(1000)
    assert result.get("success"), result.get("msg")
    return os.path.join(output_dir, "组A", "测试模板", "1.png")


def test_disabled_diversify_matches_none_output():
    root = tempfile.mkdtemp(prefix="rongjing_batch_regression_")
    try:
        template, ppt_path = make_fixture(root)
        tasks = [("组A", [ppt_path], [template])]

        none_path = run_batch(tasks, os.path.join(root, "none"), None)
        disabled = DiversifyConfig.preset("medium")
        disabled.enabled = False
        disabled_path = run_batch(tasks, os.path.join(root, "disabled"), disabled)

        assert sha256(none_path) == sha256(disabled_path)
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_enabled_diversify_changes_repeated_outputs():
    root = tempfile.mkdtemp(prefix="rongjing_batch_diversify_")
    try:
        template, ppt_path = make_fixture(root)
        tasks = [("组A", [ppt_path], [template])]
        config = DiversifyConfig.preset("medium")

        first_path = run_batch(tasks, os.path.join(root, "first"), config)
        second_path = run_batch(tasks, os.path.join(root, "second"), config)

        assert sha256(first_path) != sha256(second_path)
    finally:
        shutil.rmtree(root, ignore_errors=True)


def run_tests():
    test_disabled_diversify_matches_none_output()
    test_enabled_diversify_changes_repeated_outputs()
    print("batch runner regression tests passed")


if __name__ == "__main__":
    run_tests()
