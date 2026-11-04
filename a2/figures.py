"""Labelled image grids of MNIST digits, drawn with Pillow (no plotting library needed)."""
from __future__ import annotations

from typing import NamedTuple, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont

SCALE = 3                  # 28x28 -> 84x84, nearest neighbour (pixels are shown as-is)
PAD = 8
FG, BG, MUTED = (20, 20, 20), (255, 255, 255), (90, 90, 90)


class Tile(NamedTuple):
    image: np.ndarray      # 28x28 uint8, original MNIST pixels
    caption: Sequence[str]


class Placement(NamedTuple):
    row: int
    col: int
    box: tuple[int, int, int, int]   # (left, top, right, bottom) of the scaled image in the PNG


def _font(size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.load_default(size=size)


def make_grid(rows: Sequence[Sequence[Tile | None]], title: str,
              row_labels: Sequence[str] | None = None, caption_lines: int = 2,
              tile_width: int | None = None) -> tuple[Image.Image, list[Placement]]:
    """Grid with one list of tiles per row (None leaves a cell empty).
    Returns the image and the pixel box of every placed digit."""
    f_title, f_cap, f_row = _font(18), _font(12), _font(14)
    img_px = 28 * SCALE
    cell_w = max(tile_width or 0, img_px) + PAD
    line_h = 15
    cell_h = img_px + caption_lines * line_h + PAD
    label_w = 70 if row_labels else 0
    n_cols = max((len(r) for r in rows), default=1)
    title_h = 34
    width = PAD + label_w + n_cols * cell_w + PAD
    height = title_h + len(rows) * cell_h + PAD

    canvas = Image.new("RGB", (width, height), BG)
    draw = ImageDraw.Draw(canvas)
    draw.text((PAD, 8), title, fill=FG, font=f_title)
    placements: list[Placement] = []
    for r, row in enumerate(rows):
        top = title_h + r * cell_h
        if row_labels:
            draw.text((PAD, top + img_px // 2 - 8), row_labels[r], fill=MUTED, font=f_row)
        for c, tile in enumerate(row):
            if tile is None:
                continue
            arr = np.asarray(tile.image, dtype=np.uint8)
            if arr.shape != (28, 28):
                raise ValueError(f"expected a 28x28 image, got {arr.shape}")
            left = PAD + label_w + c * cell_w + (cell_w - PAD - img_px) // 2
            digit = Image.fromarray(arr).resize((img_px, img_px), Image.Resampling.NEAREST)
            canvas.paste(digit.convert("RGB"), (left, top))
            box = (left, top, left + img_px, top + img_px)
            placements.append(Placement(r, c, box))
            text_left = PAD + label_w + c * cell_w
            for k, line in enumerate(tile.caption[:caption_lines]):
                draw.text((text_left, top + img_px + 2 + k * line_h), line, fill=FG, font=f_cap)
    return canvas, placements


def crop_digit(png: Image.Image, box: tuple[int, int, int, int]) -> np.ndarray:
    """Recover the original 28x28 pixels from a placed tile (inverse of the nearest upscale)."""
    region = np.asarray(png.convert("L").crop(box))
    return region[::SCALE, ::SCALE]
