"""Write a resized copy of every image once, so training never decodes 16-megapixel phone photos.

Very elongated images (aspect > ``max_aspect``, e.g. the 4096x256 line-scan strips in the
defect-free class) are centre-cropped to a square first. Squashing a 16:1 strip to 224x224 would
make it trivially recognisable by its distortion alone, which is a shortcut, not defect detection.
"""
from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageOps


def square_center_crop(im: Image.Image) -> Image.Image:
    w, h = im.size
    s = min(w, h)
    left, top = (w - s) // 2, (h - s) // 2
    return im.crop((left, top, left + s, top + s))


def cache_image(src: str, dst: Path, short_side: int = 320, max_aspect: float = 2.0) -> tuple[int, int, bool]:
    with Image.open(src) as im:
        im = ImageOps.exif_transpose(im).convert("RGB")
        w, h = im.size
        cropped = max(w, h) / max(1, min(w, h)) > max_aspect
        if cropped:
            im = square_center_crop(im)
            w, h = im.size
        scale = short_side / min(w, h)
        if scale < 1:
            im = im.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.LANCZOS)
        dst.parent.mkdir(parents=True, exist_ok=True)
        im.save(dst, quality=95)
        return im.size[0], im.size[1], cropped
