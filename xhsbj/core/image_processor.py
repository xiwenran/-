from typing import List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFilter


def order_points(pts: List[List[float]]) -> np.ndarray:
    """Reorder 4 points as [TL, TR, BR, BL] regardless of input order."""
    pts = np.array(pts, dtype=np.float32)
    s = pts.sum(axis=1)
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    d = np.diff(pts, axis=1).flatten()
    tr = pts[np.argmin(d)]
    bl = pts[np.argmax(d)]
    return np.array([tl, tr, br, bl], dtype=np.float32)


def _perspective_coeffs(src_pts: np.ndarray, dst_pts: np.ndarray) -> tuple:
    """
    Compute PIL PERSPECTIVE 8 coefficients mapping dst→src.
    PIL formula:  x_in = (a*x + b*y + c) / (g*x + h*y + 1)
                  y_in = (d*x + e*y + f) / (g*x + h*y + 1)
    (x, y)       = destination (bg) coordinate
    (x_in, y_in) = source (ppt) coordinate
    """
    matrix = []
    rhs = []
    for (xd, yd), (xs, ys) in zip(dst_pts, src_pts):
        matrix.append([xd, yd, 1, 0,  0,  0, -xs * xd, -xs * yd])
        matrix.append([0,  0,  0, xd, yd,  1, -ys * xd, -ys * yd])
        rhs.extend([xs, ys])
    A = np.array(matrix, dtype=np.float64)
    b = np.array(rhs, dtype=np.float64)
    return tuple(np.linalg.solve(A, b))


def embed_image_pil(
    ppt_img: Image.Image,
    bg_img: Image.Image,
    points: List[List[float]],
    feather: int = 2,
) -> Image.Image:
    """
    Perspective-warp ppt_img into the quadrilateral defined by points on bg_img.
    feather: Gaussian blur radius applied to the mask edge for smooth blending.
    Pure PIL/numpy implementation — no cv2 dependency.
    """
    ppt_img = ppt_img.convert("RGBA")
    bg_img  = bg_img.convert("RGBA")

    bg_w, bg_h = bg_img.size
    ppt_w, ppt_h = ppt_img.size

    src_pts = np.float64([[0, 0], [ppt_w, 0], [ppt_w, ppt_h], [0, ppt_h]])
    dst_pts = order_points(points).astype(np.float64)

    # PIL PERSPECTIVE maps OUTPUT(bg) → INPUT(ppt), so dst→src coefficients
    coeffs = _perspective_coeffs(src_pts, dst_pts)
    warped = ppt_img.transform(
        (bg_w, bg_h), Image.PERSPECTIVE, coeffs, Image.BICUBIC
    )

    # Build mask for the quadrilateral region
    mask = Image.new("L", (bg_w, bg_h), 0)
    draw = ImageDraw.Draw(mask)
    poly = [(float(p[0]), float(p[1])) for p in dst_pts]
    draw.polygon(poly, fill=255)

    # Inward feathering: erode (MinFilter≈morphological erosion) then blur,
    # clip blurred result to the original hard quad boundary.
    if feather > 0:
        mask_orig = np.array(mask)
        mask = mask.filter(ImageFilter.MinFilter(3))          # 3×3 erosion
        mask = mask.filter(ImageFilter.GaussianBlur(feather)) # soften edge
        mask = Image.fromarray(
            np.where(mask_orig >= 128, np.array(mask), 0).astype(np.uint8)
        )

    # Alpha blend: result = (1 - mask) * bg + mask * warped
    bg_arr     = np.array(bg_img,  dtype=np.float32)
    warped_arr = np.array(warped,  dtype=np.float32)
    mask_f     = np.array(mask,    dtype=np.float32)[:, :, np.newaxis] / 255.0

    result = (1.0 - mask_f) * bg_arr + mask_f * warped_arr
    return Image.fromarray(result.astype(np.uint8), "RGBA")


def precompute_template_cache(
    bg_img: Image.Image,
    points: List[List[float]],
    feather: int = 2,
) -> dict:
    """Precompute mask and background array for a template.

    Call once per template, then pass the returned cache dict to
    embed_image_pil_fast() for each image/frame.  Avoids redundant mask
    computation when processing many images or video frames with the same
    template.
    """
    bg_img = bg_img.convert("RGBA")
    bg_w, bg_h = bg_img.size
    dst_pts = order_points(points).astype(np.float64)

    mask = Image.new("L", (bg_w, bg_h), 0)
    draw = ImageDraw.Draw(mask)
    poly = [(float(p[0]), float(p[1])) for p in dst_pts]
    draw.polygon(poly, fill=255)
    if feather > 0:
        mask_orig = np.array(mask)
        mask = mask.filter(ImageFilter.MinFilter(3))
        mask = mask.filter(ImageFilter.GaussianBlur(feather))
        mask = Image.fromarray(
            np.where(mask_orig >= 128, np.array(mask), 0).astype(np.uint8)
        )

    return {
        "dst_pts":  dst_pts,
        "bg_size":  (bg_w, bg_h),
        "mask_f":   np.array(mask, dtype=np.float32)[:, :, np.newaxis] / 255.0,
        "bg_arr":   np.array(bg_img, dtype=np.float32),
        # _coeffs / _coeffs_key filled lazily on first frame of a given size
    }


def embed_image_pil_fast(ppt_img: Image.Image, cache: dict) -> Image.Image:
    """Embed using a precomputed template cache (see precompute_template_cache).

    Coefficients are also cached inside `cache` keyed by ppt image size, so
    consecutive calls with the same-size input (e.g. video frames or uniform
    PPT screenshots) skip the numpy linear-solve entirely.
    """
    ppt_img = ppt_img.convert("RGBA")
    ppt_w, ppt_h = ppt_img.size

    # Lazily cache perspective coefficients per source resolution
    size_key = (ppt_w, ppt_h)
    if cache.get("_coeffs_key") != size_key:
        src_pts = np.float64([[0, 0], [ppt_w, 0], [ppt_w, ppt_h], [0, ppt_h]])
        cache["_coeffs"]     = _perspective_coeffs(src_pts, cache["dst_pts"])
        cache["_coeffs_key"] = size_key

    bg_w, bg_h = cache["bg_size"]
    warped = ppt_img.transform(
        (bg_w, bg_h), Image.PERSPECTIVE, cache["_coeffs"], Image.BICUBIC
    )

    warped_arr = np.array(warped, dtype=np.float32)
    result = (1.0 - cache["mask_f"]) * cache["bg_arr"] + cache["mask_f"] * warped_arr
    return Image.fromarray(result.astype(np.uint8), "RGBA")


def embed_image(
    ppt_path: str,
    bg_path: str,
    points: List[List[float]],
    output_size: Optional[Tuple[int, int]] = None,
    feather: int = 2,
) -> Image.Image:
    """Load from paths, embed, and optionally resize output."""
    ppt_img = Image.open(ppt_path)
    bg_img  = Image.open(bg_path)
    result  = embed_image_pil(ppt_img, bg_img, points, feather=feather)
    if output_size:
        result = result.resize(output_size, Image.LANCZOS)
    return result
