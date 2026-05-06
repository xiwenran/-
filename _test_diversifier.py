import math
import random

import numpy as np
from PIL import Image

from core.diversifier import (
    DiversifyConfig,
    diversify_image,
    jitter_points,
    strip_metadata,
)


def psnr(a: Image.Image, b: Image.Image) -> float:
    arr_a = np.array(a).astype(np.float32)
    arr_b = np.array(b).astype(np.float32)
    mse = np.mean((arr_a - arr_b) ** 2)
    if mse == 0:
        return float("inf")
    return 10 * math.log10((255**2) / mse)


def test_same_seed_is_reproducible() -> None:
    img = Image.new("RGB", (200, 200), color=(100, 150, 200))
    config = DiversifyConfig.preset("medium")

    out1 = diversify_image(img, config, seed=123)
    out2 = diversify_image(img, config, seed=123)

    assert np.array_equal(np.array(out1), np.array(out2))


def test_different_seeds_are_different() -> None:
    img = Image.new("RGB", (200, 200), color=(100, 150, 200))
    config = DiversifyConfig.preset("medium")

    out1 = diversify_image(img, config, seed=123)
    out2 = diversify_image(img, config, seed=456)

    assert not np.array_equal(np.array(out1), np.array(out2))


def test_jitter_points_stays_in_range() -> None:
    points = [[0, 0], [100, 0], [100, 100], [0, 100]]
    jitter_px = 5
    result = jitter_points(points, jitter_px, random.Random(123))

    for original, jittered in zip(points, result):
        assert abs(jittered[0] - original[0]) <= jitter_px
        assert abs(jittered[1] - original[1]) <= jitter_px


def test_medium_psnr_is_at_least_35db() -> None:
    img = Image.new("RGB", (200, 200), color=(100, 150, 200))
    config = DiversifyConfig.preset("medium")

    out = diversify_image(img, config, seed=0)

    assert psnr(img, out) >= 35


def test_strip_metadata_removes_exif() -> None:
    img = Image.new("RGB", (200, 200), color=(100, 150, 200))
    exif = Image.Exif()
    exif[271] = "Rongjing"
    img.info["exif"] = exif.tobytes()

    stripped = strip_metadata(img)

    assert len(stripped.getexif()) == 0


def run_tests() -> None:
    test_same_seed_is_reproducible()
    test_different_seeds_are_different()
    test_jitter_points_stays_in_range()
    test_medium_psnr_is_at_least_35db()
    test_strip_metadata_removes_exif()
    print("All tests passed.")


if __name__ == "__main__":
    run_tests()
