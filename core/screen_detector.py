from __future__ import annotations

import os
from typing import Optional

import cv2
import numpy as np
from PIL import Image


PointList = list[list[float]]


def detect_screen_points(image) -> Optional[PointList]:
    """Detect screen quadrilateral points in TL, TR, BR, BL order."""
    rgb = _load_rgb_array(image)
    if rgb is None:
        return None

    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    dark_ratio = float(np.mean(gray < 30))

    if dark_ratio > 0.02:
        points = _detect_solid_black_screen(gray)
        if points is not None:
            return points

    return _detect_screen_edges(gray)


def _load_rgb_array(image) -> Optional[np.ndarray]:
    try:
        if isinstance(image, Image.Image):
            pil_image = image
        elif isinstance(image, (str, os.PathLike)):
            pil_image = Image.open(image)
        else:
            return None
        return np.asarray(pil_image.convert("RGB"))
    except Exception:
        return None


def _detect_solid_black_screen(gray: np.ndarray) -> Optional[PointList]:
    h, w = gray.shape[:2]
    image_area = float(w * h)
    mask = np.where(gray < 30, 255, 0).astype(np.uint8)

    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    open_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    candidates = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < image_area * 0.01 or area > image_area * 0.95:
            continue
        points = _quad_from_contour(contour)
        if points is not None:
            candidates.append((area, points))

    if not candidates:
        return None
    return _clamp_points(max(candidates, key=lambda item: item[0])[1], w, h)


def _detect_screen_edges(gray: np.ndarray) -> Optional[PointList]:
    h, w = gray.shape[:2]
    image_area = float(w * h)

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    edge_ratio = float(np.mean(edges > 0))
    if edge_ratio > 0.25:
        return None

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edges = cv2.dilate(edges, kernel, iterations=1)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidates = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < image_area * 0.03 or area > image_area * 0.95:
            continue

        perimeter = cv2.arcLength(contour, True)
        if perimeter <= 0:
            continue
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
        if len(approx) != 4 or not cv2.isContourConvex(approx):
            continue

        points = approx.reshape(4, 2).astype(np.float32)
        quad_area = cv2.contourArea(points)
        if quad_area < image_area * 0.03:
            continue
        candidates.append((quad_area, _order_points(points)))

    if not candidates:
        return None
    return _clamp_points(max(candidates, key=lambda item: item[0])[1], w, h)


def _quad_from_contour(contour) -> Optional[PointList]:
    perimeter = cv2.arcLength(contour, True)
    if perimeter <= 0:
        return None

    approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
    if len(approx) == 4 and cv2.isContourConvex(approx):
        return _order_points(approx.reshape(4, 2).astype(np.float32))

    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect).astype(np.float32)
    box_area = cv2.contourArea(box)
    contour_area = cv2.contourArea(contour)
    if box_area <= 0 or contour_area / box_area < 0.85:
        return None
    return _order_points(box)


def _order_points(points: np.ndarray) -> PointList:
    pts = points.astype(np.float32)
    ordered = np.zeros((4, 2), dtype=np.float32)

    sums = pts.sum(axis=1)
    diffs = np.diff(pts, axis=1).reshape(-1)
    ordered[0] = pts[np.argmin(sums)]
    ordered[2] = pts[np.argmax(sums)]
    ordered[1] = pts[np.argmin(diffs)]
    ordered[3] = pts[np.argmax(diffs)]

    return [[float(x), float(y)] for x, y in ordered]


def _clamp_points(points: PointList, width: int, height: int) -> PointList:
    return [
        [
            max(0.0, min(float(width - 1), float(x))),
            max(0.0, min(float(height - 1), float(y))),
        ]
        for x, y in points
    ]
