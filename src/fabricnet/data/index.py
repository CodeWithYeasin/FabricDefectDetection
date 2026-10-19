"""Build a clean image index from one or more class-per-folder datasets.

Besides the label, every image gets provenance fields that the split logic needs:

kind
  ``original``  a photograph
  ``derived``   a file generated from another file in the same folder: ``N_processed (k).jpg``
                (binarised copies of ``N.jpg`` in the Kaggle Fabric Defects Dataset),
                ``name - Copy.png``, ``name(1).jpg``. Derived files share a ``base_key`` with their
                source and are kept in the same fold. By default they are used for training only.
  ``nonphoto``  an almost-binary image (<= 4 grey levels) that has no source photo in the folder,
                e.g. the ``exp*_num_*.png`` masks in the *Vertical* folder. Dropped by default.
base_key
  ``<folder>/<stem without derivation suffix>`` used to group a source with its derivatives.
name_label
  class implied by a filename prefix such as ``hole_2018-...`` or ``line_2018-...`` (or empty).
  A mismatch with the folder label is reported for manual checking, never silently "fixed".

The v1 pipeline treated the *dataset folder name* ``fabricfaultdetection`` as a class. Here every
folder name goes through an explicit alias table and unknown folders are reported and skipped.
"""
from __future__ import annotations

import hashlib
import re
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from .. import CLASSES

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

DEFAULT_ALIASES = {
    "defect free": "defect_free", "defect_free": "defect_free", "defectfree": "defect_free",
    "good": "defect_free", "normal": "defect_free",
    "hole": "hole", "holes": "hole",
    "horizontal": "horizontal",
    "lines": "lines", "line": "lines",
    "stain": "stain", "stains": "stain",
    "vertical": "vertical",
}

_DERIVED = re.compile(r"(?:_processed\s*\(\d+\)|\s*-\s*copy(?:\s*\(\d+\))?|\s*\(\d+\))$", re.IGNORECASE)
_PREFIX = re.compile(r"^(defect[ _]?free|hole|holes|horizontal|lines?|stains?|vertical)_", re.IGNORECASE)


def _md5(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()


def split_derived(stem: str) -> tuple[str, bool]:
    base = _DERIVED.sub("", stem).strip()
    return base, base != stem.strip()


def _grey_levels(im: Image.Image) -> int:
    g = im.convert("L")
    g.thumbnail((256, 256))
    return int(len(np.unique(np.asarray(g))))


def build_index(roots, aliases=None, classes=CLASSES, check_readable: bool = True):
    """Scan ``roots`` recursively. Label = nearest ancestor folder that matches an alias.

    Returns ``(index, skipped)``.
    """
    aliases = {k.lower(): v for k, v in (aliases or DEFAULT_ALIASES).items()}
    rows, skipped = [], []
    for root in roots:
        root = Path(root)
        for p in sorted(root.rglob("*")):
            if p.suffix.lower() not in IMG_EXT or not p.is_file():
                continue
            label = None
            for parent in p.relative_to(root).parents:
                name = parent.name.strip().lower()
                if name in aliases:
                    label = aliases[name]
                    break
            if label is None or label not in classes:
                skipped.append({"path": str(p), "reason": "no class folder"})
                continue
            w = h = levels = -1
            mode = ""
            if check_readable:
                try:
                    with Image.open(p) as im:
                        im.verify()
                    with Image.open(p) as im:
                        w, h, mode = im.size[0], im.size[1], im.mode
                        levels = _grey_levels(im)
                except Exception as e:  # corrupt file
                    skipped.append({"path": str(p), "reason": f"unreadable: {e}"})
                    continue
            base, derived = split_derived(p.stem)
            m = _PREFIX.match(p.stem)
            name_label = aliases.get(m.group(1).lower().replace("_", " "), "") if m else ""
            rows.append({
                "path": str(p), "label": label, "y": classes.index(label), "source": root.name,
                "width": w, "height": h, "mode": mode, "grey_levels": levels,
                "filesize": p.stat().st_size, "md5": _md5(p),
                "base_key": f"{p.parent.name}/{base}", "derived": derived, "name_label": name_label,
            })
    df = pd.DataFrame(rows)
    if len(df):
        has_source = set(df.loc[~df["derived"], "base_key"])
        # a derived file whose source photo is missing is treated as an original of its own
        df.loc[df["derived"] & ~df["base_key"].isin(has_source), "derived"] = False
        binary = df["grey_levels"].between(0, 4)
        df["kind"] = np.where(df["derived"], "derived", np.where(binary, "nonphoto", "original"))
    return df, pd.DataFrame(skipped, columns=["path", "reason"])
