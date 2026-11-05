"""Labelled image grids of MNIST digits, drawn with Pillow (no plotting library needed)."""
from __future__ import annotations

import math
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


# Sequential single-hue ramp (light = low, dark = high), steps 100..700 of the reference blue ramp.
BLUE_RAMP = ("#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5",
             "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b")
INK, INK_INVERSE, GAP = (20, 20, 20), (255, 255, 255), 2


def _hex(c: str) -> tuple[int, int, int]:
    return tuple(int(c[i:i + 2], 16) for i in (1, 3, 5))


def ramp_color(t: float, ramp: Sequence[str] = BLUE_RAMP) -> tuple[int, int, int]:
    """Linear interpolation along the ramp, t in [0, 1]."""
    t = min(max(t, 0.0), 1.0) * (len(ramp) - 1)
    i = min(int(t), len(ramp) - 2)
    a, b, f = _hex(ramp[i]), _hex(ramp[i + 1]), t - i
    return tuple(round(x + (y - x) * f) for x, y in zip(a, b))


def _luminance(rgb) -> float:
    def lin(c):
        c /= 255
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4
    r, g, b = (lin(v) for v in rgb)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def text_color_for(fill) -> tuple[int, int, int]:
    """Dark ink or white, whichever has the higher contrast ratio against the fill."""
    lum = _luminance(fill)
    return INK if (lum + 0.05) / (_luminance(INK) + 0.05) >= 1.05 / (lum + 0.05) else INK_INVERSE


def heatmap(values: Sequence[Sequence[float]], row_labels: Sequence[str], col_labels: Sequence[str],
            title: str, row_axis: str, col_axis: str, value_fmt: str = "{:.2f}",
            highlight: tuple[int, int] | None = None, highlight_label: str = "best",
            scale_label: str = "") -> Image.Image:
    """Annotated sequential heatmap. Colour encodes value (min..max of the data); every cell is
    also labelled with its value, and the highlighted cell gets an outline plus a text label."""
    f_title, f_axis, f_cell, f_small = _font(17), _font(14), _font(15), _font(12)
    n_rows, n_cols = len(values), len(values[0])
    cell_w, cell_h, left, top = 96, 54, 130, 96
    flat = [v for row in values for v in row]
    lo, hi = min(flat), max(flat)
    span = (hi - lo) or 1.0
    grid_w, grid_h = n_cols * cell_w, n_rows * cell_h
    legend_top = top + grid_h + 26
    width, height = left + grid_w + 30, legend_top + 64
    img = Image.new("RGB", (width, height), BG)
    d = ImageDraw.Draw(img)
    d.text((16, 14), title, fill=FG, font=f_title)
    d.text((left + grid_w // 2, top - 34), col_axis, fill=MUTED, font=f_axis, anchor="mb")
    d.text((left - 10, top - 6), row_axis, fill=MUTED, font=f_axis, anchor="rb")
    for c, lab in enumerate(col_labels):
        d.text((left + c * cell_w + cell_w // 2, top - 6), lab, fill=FG, font=f_axis, anchor="mb")
    for r, row in enumerate(values):
        y0 = top + r * cell_h
        d.text((left - 10, y0 + cell_h // 2), row_labels[r], fill=FG, font=f_axis, anchor="rm")
        for c, v in enumerate(row):
            x0 = left + c * cell_w
            fill = ramp_color((v - lo) / span)
            box = (x0 + GAP // 2, y0 + GAP // 2, x0 + cell_w - GAP // 2 - 1, y0 + cell_h - GAP // 2 - 1)
            d.rounded_rectangle(box, radius=4, fill=fill)
            ink = text_color_for(fill)
            label = value_fmt.format(v)
            if highlight == (r, c):
                d.rounded_rectangle(box, radius=4, outline=FG, width=3)
                inner = (box[0] + 3, box[1] + 3, box[2] - 3, box[3] - 3)
                d.rounded_rectangle(inner, radius=3, outline=BG, width=2)
                d.text((x0 + cell_w // 2, y0 + cell_h // 2 - 8), label, fill=ink, font=f_cell, anchor="mm")
                d.text((x0 + cell_w // 2, y0 + cell_h // 2 + 11), highlight_label, fill=ink, font=f_small,
                       anchor="mm")
            else:
                d.text((x0 + cell_w // 2, y0 + cell_h // 2), label, fill=ink, font=f_cell, anchor="mm")
    # scale legend: gradient bar with min / max labels
    bar_l, bar_w, bar_h = left, grid_w, 12
    for x in range(bar_w):
        d.line([(bar_l + x, legend_top), (bar_l + x, legend_top + bar_h)], fill=ramp_color(x / (bar_w - 1)))
    d.text((bar_l, legend_top + bar_h + 6), value_fmt.format(lo), fill=FG, font=f_small, anchor="lt")
    d.text((bar_l + bar_w, legend_top + bar_h + 6), value_fmt.format(hi), fill=FG, font=f_small, anchor="rt")
    if scale_label:
        d.text((bar_l + bar_w // 2, legend_top + bar_h + 6), scale_label, fill=MUTED, font=f_small, anchor="mt")
    return img


# ---- Question 8 charts (Pillow; one hue for data, gray for context, text in ink colours) ----
EMPHASIS, CONTEXT, GRID, INK_2 = (42, 120, 214), (150, 150, 150), (228, 228, 228), (85, 85, 85)


def _nice_ticks(lo: float, hi: float, n: int = 5) -> list[float]:
    raw = (hi - lo) / n
    mag = 10 ** math.floor(math.log10(raw))
    step = min((s * mag for s in (1, 2, 2.5, 5, 10) if s * mag >= raw), default=raw)
    start = math.ceil(lo / step) * step
    return [round(start + i * step, 10) for i in range(int((hi - start) / step + 1e-9) + 1)]


def dot_plot(rows: Sequence[tuple[str, float, float, bool]], title: str, subtitle: str, x_label: str,
             x_min: float, x_max: float, value_fmt: str = "{:.2f} ± {:.2f}") -> Image.Image:
    """Horizontal dot plot with ±SD whiskers. rows = (label, mean, sd, emphasised), drawn top to bottom."""
    f_title, f_sub, f_lab, f_small = _font(18), _font(13), _font(14), _font(12)
    left, right, top, row_h = 300, 150, 78, 40
    plot_w = 520
    width, height = left + plot_w + right, top + row_h * len(rows) + 62
    img = Image.new("RGB", (width, height), BG)
    d = ImageDraw.Draw(img)
    d.text((16, 12), title, fill=FG, font=f_title)
    d.text((16, 40), subtitle, fill=INK_2, font=f_sub)
    x = lambda v: left + (v - x_min) / (x_max - x_min) * plot_w
    bottom = top + row_h * len(rows)
    for t in _nice_ticks(x_min, x_max):
        d.line([(x(t), top - 6), (x(t), bottom)], fill=GRID, width=1)
        d.text((x(t), bottom + 6), f"{t:g}", fill=INK_2, font=f_small, anchor="mt")
    d.text((left + plot_w // 2, bottom + 28), x_label, fill=INK_2, font=f_lab, anchor="mt")
    for i, (label, mean, sd, emph) in enumerate(rows):
        y = top + row_h * i + row_h // 2
        colour = EMPHASIS if emph else CONTEXT
        d.text((left - 14, y), label, fill=FG, font=f_lab, anchor="rm")
        lo, hi = max(mean - sd, x_min), min(mean + sd, x_max)
        d.line([(x(lo), y), (x(hi), y)], fill=colour, width=2)
        for e in (lo, hi):
            d.line([(x(e), y - 6), (x(e), y + 6)], fill=colour, width=2)
        r = 6
        d.ellipse([x(mean) - r - 2, y - r - 2, x(mean) + r + 2, y + r + 2], fill=BG)      # 2px surface ring
        d.ellipse([x(mean) - r, y - r, x(mean) + r, y + r], fill=colour)
        d.text((x(hi) + 12, y), value_fmt.format(mean, sd), fill=FG if emph else INK_2, font=f_small, anchor="lm")
    return img


def bar_panels(labels: Sequence[str], panels: Sequence[tuple[str, Sequence[float], str]], title: str,
               subtitle: str) -> Image.Image:
    """Small multiples of horizontal bars sharing the row labels; each panel has its own zero-based axis.
    panels = (panel title, values, value format)."""
    f_title, f_sub, f_lab, f_small = _font(18), _font(13), _font(14), _font(12)
    left, top, row_h, bar_h, panel_w, gap = 300, 96, 40, 18, 330, 40
    width = left + len(panels) * (panel_w + gap) + 20
    height = top + row_h * len(labels) + 56
    img = Image.new("RGB", (width, height), BG)
    d = ImageDraw.Draw(img)
    d.text((16, 12), title, fill=FG, font=f_title)
    d.text((16, 40), subtitle, fill=INK_2, font=f_sub)
    bottom = top + row_h * len(labels)
    for i, lab in enumerate(labels):
        d.text((left - 14, top + row_h * i + row_h // 2), lab, fill=FG, font=f_lab, anchor="rm")
    for k, (ptitle, values, fmt) in enumerate(panels):
        x0 = left + k * (panel_w + gap)
        vmax = max(values)
        ticks = _nice_ticks(0, vmax * 1.05)
        scale = (panel_w - 70) / max(ticks[-1], vmax)
        d.text((x0, top - 26), ptitle, fill=FG, font=f_lab)
        for t in ticks:
            d.line([(x0 + t * scale, top - 4), (x0 + t * scale, bottom)], fill=GRID, width=1)
            d.text((x0 + t * scale, bottom + 6), f"{t:g}", fill=INK_2, font=f_small, anchor="mt")
        d.line([(x0, top - 4), (x0, bottom)], fill=CONTEXT, width=1)                     # baseline
        for i, v in enumerate(values):
            y = top + row_h * i + (row_h - bar_h) // 2
            w = max(v * scale, 2)
            d.rounded_rectangle([x0, y, x0 + w, y + bar_h], radius=4, fill=EMPHASIS)
            d.rectangle([x0, y, x0 + min(4, w), y + bar_h], fill=EMPHASIS)                # square at the baseline
            d.text((x0 + w + 6, y + bar_h // 2), fmt.format(v), fill=FG, font=f_small, anchor="lm")
    return img
