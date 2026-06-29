"""Render a fake daily-rainfall document as a matplotlib Figure.

The rendered image mimics a pre-printed form with typewritten rainfall
values: off-white paper, dark grid lines, month names across the top,
day numbers down the left, and individual cell values.
"""

from __future__ import annotations

import random

import matplotlib
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from . import geometry as geo
from . import jitter as jtr
from .constants import MONTHS, get_available_font_families

matplotlib.use("Agg")

# ── Constants ──────────────────────────────────────────────────────────────────

# Base background color (before per-page jitter). Used to set grid line
# intensity: when line_intensity is low, grid_ink approaches this value so
# lines blend into the background and become invisible.
BACKGROUND_COLOR_BASE = 0.94

# ── Internal helpers ──────────────────────────────────────────────────────────


def _ink(g: float) -> tuple[float, float, float]:
    """Return an RGB tuple for a gray level *g* (0 = black, 1 = white)."""
    return (g, g, g)


def _tx(x: float, scale_x: float, shift_x: float) -> float:
    """Apply horizontal table scaling and translation around grid centre."""
    cx = (geo.GRID_LEFT + geo.GRID_RIGHT) / 2
    return cx + (x - cx) * scale_x + shift_x


def _ty(y: float, scale_y: float, shift_y: float) -> float:
    """Apply vertical table scaling and translation around grid centre."""
    cy = (geo.GRID_TOP + geo.GRID_BOTTOM) / 2
    return cy + (y - cy) * scale_y + shift_y


def _txy(
    x: float,
    y: float,
    scale_x: float,
    scale_y: float,
    shift_x: float,
    shift_y: float,
) -> tuple[float, float]:
    """Apply 2D table transform (scale + translation)."""
    return _tx(x, scale_x, shift_x), _ty(y, scale_y, shift_y)


def _grid_line_x(col: int, n_cols: int) -> float:
    """X position of a vertical grid boundary for a dynamic column count."""
    return geo.GRID_LEFT + col * (geo.GRID_W / n_cols)


def _cell_center(row: int, col: int, n_cols: int) -> tuple[float, float]:
    """Centre (x, y) of a cell for a dynamic column count."""
    x = geo.GRID_LEFT + (col + 0.5) * (geo.GRID_W / n_cols)
    y = geo.GRID_TOP - (row + 0.5) * geo.CELL_H
    return x, y


def _draw_background(ax: plt.Axes, rng: np.random.Generator) -> None:
    """Fill the page with a slightly jittered off-white background."""
    bg = jtr.gray(BACKGROUND_COLOR_BASE, rng, sigma=0.015)
    ax.add_patch(mpatches.Rectangle((0, 0), 1, 1, color=_ink(bg), zorder=0))


def _draw_grid_lines(
    ax: plt.Axes,
    rng: np.random.Generator,
    ink: float,
    jitter_pts: float,
    scale_x: float,
    scale_y: float,
    shift_x: float,
    shift_y: float,
    n_cols: int,
    base_line_intensity: float,
    individual_line_intensity_sigma: float,
) -> None:
    """Draw all vertical and horizontal grid lines with slight positional jitter."""
    border_lw = jtr.linewidth(1.6, rng, 0.15)
    inner_lw = jtr.linewidth(0.8, rng, 0.08)

    # Vertical lines (one per column boundary, plus right edge)
    for ci in range(n_cols + 1):
        # Per-line intensity variation, additive to baseline.
        line_intensity_adjust = float(
            np.clip(rng.normal(0, individual_line_intensity_sigma), -1.0, 1.0)
        )
        current_line_intensity = np.clip(
            base_line_intensity + line_intensity_adjust, 0.0, 1.0
        )
        current_grid_ink = BACKGROUND_COLOR_BASE - current_line_intensity * 0.86
        col = _ink(current_grid_ink)

        base_x = _tx(_grid_line_x(ci, n_cols), scale_x, shift_x)
        base_y0 = _ty(geo.GRID_BOTTOM, scale_y, shift_y)
        base_y1 = _ty(geo.GRID_TOP, scale_y, shift_y)
        x = jtr.pos(base_x, rng, jitter_pts)
        y0 = jtr.pos(base_y0, rng, jitter_pts)
        y1 = jtr.pos(base_y1, rng, jitter_pts)
        lw = border_lw if ci in (0, n_cols) else inner_lw
        ax.add_line(mlines.Line2D([x, x], [y0, y1], color=col, linewidth=lw, zorder=2))

    # Horizontal lines (one per row boundary, plus bottom edge)
    for ri in range(geo.N_ROWS + 1):
        # Per-line intensity variation, additive to baseline.
        line_intensity_adjust = float(
            np.clip(rng.normal(0, individual_line_intensity_sigma), -1.0, 1.0)
        )
        current_line_intensity = np.clip(
            base_line_intensity + line_intensity_adjust, 0.0, 1.0
        )
        current_grid_ink = BACKGROUND_COLOR_BASE - current_line_intensity * 0.86
        col = _ink(current_grid_ink)

        base_y = _ty(geo.grid_line_y(ri), scale_y, shift_y)
        base_x0 = _tx(geo.GRID_LEFT, scale_x, shift_x)
        base_x1 = _tx(geo.GRID_RIGHT, scale_x, shift_x)
        y = jtr.pos(base_y, rng, jitter_pts)
        x0 = jtr.pos(base_x0, rng, jitter_pts)
        x1 = jtr.pos(base_x1, rng, jitter_pts)
        # Draw the line separating header/data and data/totals slightly heavier
        lw = border_lw if ri in (0, 1, geo.N_ROWS - 1, geo.N_ROWS) else inner_lw
        ax.add_line(mlines.Line2D([x0, x1], [y, y], color=col, linewidth=lw, zorder=2))


def _draw_month_headers(
    ax: plt.Axes,
    rng: np.random.Generator,
    ink: float,
    font_family: str,
    font_size: float,
    jitter_pts: float,
    scale_x: float,
    scale_y: float,
    shift_x: float,
    shift_y: float,
    n_cols: int,
) -> None:
    """Render month abbreviations in the header row (row 0, cols 1–12)."""
    col = _ink(jtr.gray(ink, rng, 0.03))
    for mi, month in enumerate(MONTHS):
        cx, cy = _cell_center(0, mi + 1, n_cols)  # col 0 is left label column
        cx, cy = _txy(cx, cy, scale_x, scale_y, shift_x, shift_y)
        ax.text(
            jtr.pos(cx, rng, jitter_pts),
            jtr.pos(cy, rng, jitter_pts),
            month,
            ha="center",
            va="center",
            fontfamily=font_family,
            fontsize=jtr.size(font_size, rng, 0.4),
            rotation=jtr.rotation(rng, 0.4),
            color=col,
            zorder=3,
        )


def _draw_day_labels(
    ax: plt.Axes,
    rng: np.random.Generator,
    ink: float,
    font_family: str,
    font_size: float,
    jitter_pts: float,
    scale_x: float,
    scale_y: float,
    shift_x: float,
    shift_y: float,
    n_cols: int,
    include_right_day_labels: bool,
) -> None:
    """Render day numbers in the left label column and optional right column."""
    col = _ink(jtr.gray(ink, rng, 0.03))
    label_size = jtr.size(font_size * 0.9, rng, 0.3)

    # Day numbers: rows 1–31
    for day in range(1, 32):
        cx, cy = _cell_center(day, 0, n_cols)
        cx, cy = _txy(cx, cy, scale_x, scale_y, shift_x, shift_y)
        ax.text(
            jtr.pos(cx, rng, jitter_pts),
            jtr.pos(cy, rng, jitter_pts),
            str(day),
            ha="center",
            va="center",
            fontfamily=font_family,
            fontsize=jtr.size(label_size, rng, 0.2),
            rotation=jtr.rotation(rng, 0.3),
            color=col,
            zorder=3,
        )

        if include_right_day_labels:
            rcx, rcy = _cell_center(day, n_cols - 1, n_cols)
            rcx, rcy = _txy(rcx, rcy, scale_x, scale_y, shift_x, shift_y)
            ax.text(
                jtr.pos(rcx, rng, jitter_pts),
                jtr.pos(rcy, rng, jitter_pts),
                str(day),
                ha="center",
                va="center",
                fontfamily=font_family,
                fontsize=jtr.size(label_size, rng, 0.2),
                rotation=jtr.rotation(rng, 0.3),
                color=col,
                zorder=3,
            )

    # "Total" label in the last row
    cx, cy = _cell_center(geo.N_ROWS - 1, 0, n_cols)
    cx, cy = _txy(cx, cy, scale_x, scale_y, shift_x, shift_y)
    ax.text(
        jtr.pos(cx, rng, jitter_pts),
        jtr.pos(cy, rng, jitter_pts),
        "Total",
        ha="center",
        va="center",
        fontfamily=font_family,
        fontsize=jtr.size(font_size * 0.8, rng, 0.3),
        rotation=jtr.rotation(rng, 0.3),
        color=col,
        zorder=3,
    )


def _fmt_value(val_str: str, rng: np.random.Generator) -> str:
    """Format a value string for rendering; occasionally drop leading zero."""
    if val_str == "null":
        return ""
    # ~20% of the time render ".37" instead of "0.37" (older style)
    if val_str.startswith("0.") and rng.random() < 0.2:
        return val_str[1:]  # drop the leading "0"
    return val_str


def _draw_data_values(
    ax: plt.Axes,
    data: dict[str, list[str]],
    rng: np.random.Generator,
    ink: float,
    font_family: str,
    font_size: float,
    jitter_pts: float,
    scale_x: float,
    scale_y: float,
    shift_x: float,
    shift_y: float,
    n_cols: int,
) -> None:
    """Render daily rainfall values in their grid cells (rows 1–31, cols 1–12)."""
    for day in range(1, 32):
        row_values = data.get(f"Day {day}", ["null"] * 12)
        for mi, val_str in enumerate(row_values):
            text = _fmt_value(val_str, rng)
            if not text:
                continue
            cx, cy = _cell_center(day, mi + 1, n_cols)
            cx, cy = _txy(cx, cy, scale_x, scale_y, shift_x, shift_y)
            col = _ink(jtr.gray(ink * 0.8, rng, 0.04))
            ax.text(
                jtr.pos(cx, rng, jitter_pts),
                jtr.pos(cy, rng, jitter_pts),
                text,
                ha="center",
                va="center",
                fontfamily=font_family,
                fontsize=jtr.size(font_size * 0.85, rng, 0.3),
                rotation=jtr.rotation(rng, 0.25),
                color=col,
                zorder=3,
            )


def _draw_totals(
    ax: plt.Axes,
    totals: list[str],
    rng: np.random.Generator,
    ink: float,
    font_family: str,
    font_size: float,
    jitter_pts: float,
    scale_x: float,
    scale_y: float,
    shift_x: float,
    shift_y: float,
    n_cols: int,
) -> None:
    """Render monthly totals in the bottom row (row N_ROWS-1, cols 1–12)."""
    row = geo.N_ROWS - 1
    for mi, val_str in enumerate(totals):
        text = _fmt_value(val_str, rng)
        if not text:
            continue
        cx, cy = _cell_center(row, mi + 1, n_cols)
        cx, cy = _txy(cx, cy, scale_x, scale_y, shift_x, shift_y)
        col = _ink(jtr.gray(ink * 0.8, rng, 0.04))
        ax.text(
            jtr.pos(cx, rng, jitter_pts),
            jtr.pos(cy, rng, jitter_pts),
            text,
            ha="center",
            va="center",
            fontfamily=font_family,
            fontsize=jtr.size(font_size * 0.85, rng, 0.3),
            rotation=jtr.rotation(rng, 0.25),
            color=col,
            zorder=3,
        )


def _draw_page_header(
    ax: plt.Axes,
    year: int,
    county: str,
    station_id: int,
    rng: np.random.Generator,
    ink: float,
    font_family: str,
    font_size: float,
    shift_x: float,
    shift_y: float,
) -> None:
    """Render the station title block above the grid."""
    hx, hy = geo.header_center()
    hx += shift_x
    hy += shift_y * 0.6
    col = _ink(jtr.gray(ink, rng, 0.03))

    # Main title line
    ax.text(
        hx,
        hy + 0.025,
        f"DAILY RAINFALL  {year}",
        ha="center",
        va="center",
        fontfamily=font_family,
        fontsize=jtr.size(font_size * 1.3, rng, 0.5),
        rotation=jtr.rotation(rng, 0.2),
        color=col,
        fontweight="bold",
        zorder=3,
    )
    # Sub-title line: county + station number
    ax.text(
        hx,
        hy - 0.020,
        f"{county}   No. {station_id}",
        ha="center",
        va="center",
        fontfamily=font_family,
        fontsize=jtr.size(font_size * 1.0, rng, 0.4),
        rotation=jtr.rotation(rng, 0.2),
        color=col,
        zorder=3,
    )


# ── Public API ────────────────────────────────────────────────────────────────


def draw_document(
    data: dict[str, list[str]],
    year: int,
    county: str,
    station_id: int,
    *,
    font_family: str | None = None,
    font_size: float = 20.0,
    jitter_grid_points: float = 0.0008,
    table_scale_x: float | None = None,
    table_scale_y: float | None = None,
    table_shift_x: float | None = None,
    table_shift_y: float | None = None,
    include_right_day_labels: bool | None = None,
    right_day_label_probability: float = 0.9,
    include_post_dec_blank_column: bool | None = None,
    post_dec_blank_column_probability: float = 0.2,
    line_intensity_sigma: float = 0.40,
    individual_line_intensity_sigma: float = 0.0,
    rng: np.random.Generator | None = None,
) -> plt.Figure:
    """Render *data* as a synthetic historical daily-rainfall document.

    Parameters
    ----------
    data:
        Dict produced by :func:`make_data.generate_data`.
    year:
        Calendar year shown in the page header.
    county:
        County name shown in the page header.
    station_id:
        Numeric station ID shown in the page header.
    font_family:
        Matplotlib font family string.  Chosen at random from
        :data:`constants.FONT_FAMILIES` when *None*.
    font_size:
        Base font size in points; individual elements are scaled from this.
    jitter_grid_points:
        Std-dev of position jitter applied to grid lines and text (normalised
        page coordinates).
    table_scale_x, table_scale_y:
        Optional table scale factors. If omitted, random values are drawn
        per image from [0.95, 1.0] (slight shrink only).
    table_shift_x, table_shift_y:
        Optional table offsets in normalised page coordinates. If omitted,
        random values are drawn per image from small ranges.
    include_right_day_labels:
        Whether to render an additional right-hand day-label column.
        If omitted, it is sampled per image using *right_day_label_probability*.
    right_day_label_probability:
        Probability of including the additional right-hand day-label column
        when *include_right_day_labels* is not explicitly set.
    include_post_dec_blank_column:
        Whether to include an additional blank column immediately to the right
        of Dec. If omitted, it is sampled per image using
        *post_dec_blank_column_probability*.
    post_dec_blank_column_probability:
        Probability of including the additional blank column immediately right
        of Dec when *include_post_dec_blank_column* is not explicitly set.
    line_intensity_sigma:
        Std-dev of per-page grid line intensity jitter. Line intensity is
        sampled as ``clip(N(0.92, line_intensity_sigma), 0, 1)`` and then
        converted to grayscale as ``gray = background - intensity * 0.86``. With the default
        0.40, there is about a 1% chance intensity clips to 0 (near-invisible
        lines). This affects grid lines only; text colour is sampled
        separately.
    individual_line_intensity_sigma:
        Std-dev of per-line intensity variation added to the per-page baseline.
        Each line is sampled as ``clip(N(0, individual_line_intensity_sigma), -1, 1)``
        and added to the per-page line_intensity before conversion to grayscale.
        Default 0.0 (no per-line variation); increase for additional variation.
    rng:
        NumPy random generator.  A new default generator is used when *None*.

    Returns
    -------
    matplotlib.figure.Figure
        The rendered figure.  Call ``fig.savefig(path, ...)`` to persist it.
    """
    if rng is None:
        rng = np.random.default_rng()
    if font_family is None:
        font_family = random.choice(get_available_font_families())
    if table_scale_x is None:
        table_scale_x = float(rng.uniform(0.95, 1.0))
    if table_scale_y is None:
        table_scale_y = float(rng.uniform(0.95, 1.0))
    if table_shift_x is None:
        table_shift_x = float(rng.uniform(-0.015, 0.015))
    if table_shift_y is None:
        table_shift_y = float(rng.uniform(-0.02, 0.02))
    if include_right_day_labels is None:
        include_right_day_labels = bool(rng.random() < right_day_label_probability)
    if include_post_dec_blank_column is None:
        include_post_dec_blank_column = bool(
            rng.random() < post_dec_blank_column_probability
        )

    # Split random streams so line-intensity controls cannot influence text.
    line_rng = np.random.default_rng(rng.integers(0, 2**63 - 1))
    text_rng = np.random.default_rng(rng.integers(0, 2**63 - 1))
    bg_rng = np.random.default_rng(rng.integers(0, 2**63 - 1))

    # Left label + 12 month data columns + optional blank post-Dec column
    # + optional right day-label column.
    n_cols = (
        1
        + geo.N_DATA_COLS
        + (1 if include_post_dec_blank_column else 0)
        + (1 if include_right_day_labels else 0)
    )

    line_intensity = float(
        np.clip(line_rng.normal(0.92, line_intensity_sigma), 0.0, 1.0)
    )
    # Map line_intensity [0, 1] to grid_ink [background, 0.08 (dark)].
    # Low intensity → invisible (matches background); high intensity → visible dark lines.
    grid_ink = BACKGROUND_COLOR_BASE - line_intensity * 0.86
    text_ink = jtr.gray(0.08, text_rng, sigma=0.04)

    fig = plt.figure(
        figsize=(geo.PAGE_WIDTH_IN, geo.PAGE_HEIGHT_IN),
        dpi=geo.DPI,
    )
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("auto")
    ax.axis("off")

    _draw_background(ax, bg_rng)
    _draw_grid_lines(
        ax,
        line_rng,
        grid_ink,
        jitter_grid_points,
        table_scale_x,
        table_scale_y,
        table_shift_x,
        table_shift_y,
        n_cols,
        line_intensity,
        individual_line_intensity_sigma,
    )
    _draw_month_headers(
        ax,
        text_rng,
        text_ink,
        font_family,
        font_size,
        jitter_grid_points,
        table_scale_x,
        table_scale_y,
        table_shift_x,
        table_shift_y,
        n_cols,
    )
    _draw_day_labels(
        ax,
        text_rng,
        text_ink,
        font_family,
        font_size,
        jitter_grid_points,
        table_scale_x,
        table_scale_y,
        table_shift_x,
        table_shift_y,
        n_cols,
        include_right_day_labels,
    )
    _draw_data_values(
        ax,
        data,
        text_rng,
        text_ink,
        font_family,
        font_size,
        jitter_grid_points,
        table_scale_x,
        table_scale_y,
        table_shift_x,
        table_shift_y,
        n_cols,
    )
    _draw_totals(
        ax,
        data.get("Totals", ["null"] * 12),
        text_rng,
        text_ink,
        font_family,
        font_size,
        jitter_grid_points,
        table_scale_x,
        table_scale_y,
        table_shift_x,
        table_shift_y,
        n_cols,
    )
    _draw_page_header(
        ax,
        year,
        county,
        station_id,
        text_rng,
        text_ink,
        font_family,
        font_size,
        table_shift_x,
        table_shift_y,
    )

    return fig
