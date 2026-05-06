from PIL import Image
import numpy as np

from core.screen_detector import detect_screen_points


EXPECTED_POINTS = [[50, 40], [250, 40], [250, 160], [50, 160]]


def _max_point_delta(points, expected):
    return max(
        max(abs(points[i][0] - expected[i][0]), abs(points[i][1] - expected[i][1]))
        for i in range(4)
    )


def test_detects_solid_black_rectangle():
    img = Image.new("RGB", (300, 200), "white")
    pixels = img.load()
    for y in range(40, 161):
        for x in range(50, 251):
            pixels[x, y] = (0, 0, 0)

    points = detect_screen_points(img)
    delta = _max_point_delta(points, EXPECTED_POINTS) if points else None
    print("test_detects_solid_black_rectangle:", points, "max_delta=", delta)
    assert points is not None
    assert delta <= 5


def test_random_noise_returns_none():
    rng = np.random.default_rng(1234)
    noise = rng.integers(0, 256, size=(240, 320, 3), dtype=np.uint8)
    img = Image.fromarray(noise, "RGB")

    points = detect_screen_points(img)
    print("test_random_noise_returns_none:", points)
    assert points is None


def test_point_order_is_tl_tr_br_bl():
    img = Image.new("RGB", (300, 200), "white")
    pixels = img.load()
    for y in range(40, 161):
        for x in range(50, 251):
            pixels[x, y] = (0, 0, 0)

    points = detect_screen_points(img)
    print("test_point_order_is_tl_tr_br_bl:", points)
    assert points is not None
    for got, expected in zip(points, EXPECTED_POINTS):
        assert abs(got[0] - expected[0]) <= 5
        assert abs(got[1] - expected[1]) <= 5


if __name__ == "__main__":
    test_detects_solid_black_rectangle()
    test_random_noise_returns_none()
    test_point_order_is_tl_tr_br_bl()
    print("all screen detector tests passed")
