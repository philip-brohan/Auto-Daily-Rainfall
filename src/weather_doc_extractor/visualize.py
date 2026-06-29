"""Diagnostic figure: original image (left) + extracted-data table (right).

Public API
----------
make_figure(image_path, grid, ground_truth=None, title=None, model_name=None)
    Build and return a ``matplotlib.figure.Figure``.

save_figure(image_path, grid, output_path, ground_truth=None, title=None, model_name=None)
    Build the figure and write it to *output_path* (PNG by default).

make_comparison_figure(image_path, predicted, ground_truth, title=None, model_name=None, tolerance=0.005)
    Build a three-panel figure: source image | predicted grid | ground-truth grid.
    Predicted cells are colour-coded (blue = match, red = mismatch).

save_comparison_figure(image_path, predicted, ground_truth, output_path, ...)
    Build the three-panel figure and write it to *output_path* (PNG by default).
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from matplotlib.figure import Figure

    from weather_doc_extractor.schemas import DailyRainfallGrid

# Month abbreviations for column headers
_MONTH_ABBREVS = [
    "Jan",
    "Feb",
    "Mar",
    "Apr",
    "May",
    "Jun",
    "Jul",
    "Aug",
    "Sep",
    "Oct",
    "Nov",
    "Dec",
]

_ROW_LABELS = [f"Day {i}" for i in range(1, 32)] + ["Totals"]


def _cell_text(grid: "DailyRainfallGrid") -> list[list[str]]:
    """Convert a grid to a 32-row × 12-col list of display strings."""
    rows: list[list[str]] = []
    for label in _ROW_LABELS[:-1]:  # Day 1 … Day 31
        vals = grid.days.get(label, [None] * 12)
        rows.append([_fmt(v) for v in vals])
    rows.append([_fmt(v) for v in grid.totals])  # Totals
    return rows


def _fmt(v: float | None) -> str:
    if v is None:
        return "null"
    return f"{v:.2f}" if v != int(v) else f"{v:.0f}"


def _diff_text_colours(
    predicted: "DailyRainfallGrid",
    ground_truth: "DailyRainfallGrid",
    tolerance: float = 0.005,
) -> list[list[str]]:
    """Return a 32×12 grid of text colours for data cells.

    Blue  = match within tolerance (or both None)
    Red   = mismatch or missing prediction
    """
    colours: list[list[str]] = []
    all_pred = [predicted.days.get(f"Day {i}", [None] * 12) for i in range(1, 32)]
    all_pred.append(predicted.totals)
    all_gt = [ground_truth.days.get(f"Day {i}", [None] * 12) for i in range(1, 32)]
    all_gt.append(ground_truth.totals)

    for pred_row, gt_row in zip(all_pred, all_gt):
        row_colours: list[str] = []
        for p, g in zip(pred_row, gt_row):
            if p is None and g is None:
                row_colours.append("blue")
            elif p is None or g is None:
                row_colours.append("red")
            elif abs(p - g) <= tolerance:
                row_colours.append("blue")
            else:
                row_colours.append("red")
        colours.append(row_colours)
    return colours


def make_figure(
    image_path: Path,
    grid: "DailyRainfallGrid",
    ground_truth: "DailyRainfallGrid | None" = None,
    title: str | None = None,
    tolerance: float = 0.005,
    model_name: str | None = None,
) -> "Figure":
    """Return a matplotlib Figure with the source image and extracted table.

    Both panels are sized to match the aspect ratio of the source image so
    that the table is portrait-shaped like the original document.

    Parameters
    ----------
    image_path:
        Path to the document image.
    grid:
        The extracted (predicted) rainfall grid.
    ground_truth:
        Optional ground-truth grid.  When provided, cells are coloured
        green (match within *tolerance*) or red (mismatch).
    title:
        Figure suptitle.  Defaults to the image filename stem.
    tolerance:
        Absolute tolerance in inches for the green/red colouring.
    model_name:
        Name of the model used for extraction.  Shown as the table panel
        title.  Defaults to ``"Training data"`` when ``None``.
    """
    import matplotlib

    matplotlib.use("Agg")  # non-interactive backend; safe for scripts
    import matplotlib.pyplot as plt
    from PIL import Image as PILImage

    img = PILImage.open(image_path).convert("RGB")
    img_w, img_h = img.size
    img_aspect = img_w / img_h  # ≈ 0.63 for these portrait scans

    # Each panel is panel_h inches tall; its width matches the image aspect.
    panel_h = 11.0  # inches
    panel_w = panel_h * img_aspect
    title_margin = 0.4  # inches for suptitle
    h_gap = 0.8  # inches between panels
    l_margin = 0.2
    r_margin = 0.2
    fig_w = l_margin + panel_w + h_gap + panel_w + r_margin
    fig_h = panel_h + title_margin

    fig = plt.figure(figsize=(fig_w, fig_h))
    title = title or Path(image_path).stem
    fig.suptitle(title, fontsize=20, y=0.99)

    # Normalised axes coordinates
    bottom = 0.01
    ax_h = panel_h / fig_h - 0.02
    left_panel_left = l_margin / fig_w
    panel_w_frac = panel_w / fig_w
    right_panel_left = (l_margin + panel_w + h_gap) / fig_w

    # Table panel is 18% shorter than the image panel: top moves down,
    # bottom stays at the same position.
    tbl_ax_h = ax_h * 0.82

    # Further reduce table height by 7% of its current height: foot moves up,
    # top stays fixed.  Top = bottom + tbl_ax_h; new bottom = bottom + 0.07*tbl_ax_h.
    tbl_bottom = bottom + tbl_ax_h * 0.07
    tbl_ax_h = tbl_ax_h * 0.93

    # ── Left panel: source image ──────────────────────────────────────────
    ax_img = fig.add_axes([left_panel_left, bottom, panel_w_frac, ax_h])
    ax_img.imshow(img)
    ax_img.axis("off")
    # No title on the image panel

    # ── Right panel: table ────────────────────────────────────────────────
    ax_tbl = fig.add_axes([right_panel_left, tbl_bottom, panel_w_frac, tbl_ax_h])
    ax_tbl.axis("off")

    cell_text = _cell_text(grid)

    table_title = model_name if model_name is not None else "Training data"

    cell_colours = [["white"] * 12] * 32
    text_colours: list[list[str]] | None = None

    if ground_truth is not None:
        text_colours = _diff_text_colours(grid, ground_truth, tolerance)
        ax_tbl.set_title(table_title, fontsize=18)
    else:
        ax_tbl.set_title(table_title, fontsize=18)

    tbl = ax_tbl.table(
        cellText=cell_text,
        rowLabels=_ROW_LABELS,
        colLabels=_MONTH_ABBREVS,
        cellColours=cell_colours,
        cellLoc="center",
        loc="center",
        bbox=[0, 0, 1, 1],  # fill the full axes rectangle
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(7)
    for (row, col), cell in tbl.get_celld().items():
        cell.set_edgecolor("#aaaaaa")
        if row > 0 and col >= 0:
            cell.set_fontsize(10.5)  # 7 * 1.5
            if text_colours is not None:
                cell.get_text().set_color(text_colours[row - 1][col])
            else:
                cell.get_text().set_color("blue")

    return fig


def save_figure(
    image_path: Path,
    grid: "DailyRainfallGrid",
    output_path: Path,
    ground_truth: "DailyRainfallGrid | None" = None,
    title: str | None = None,
    tolerance: float = 0.005,
    dpi: int = 150,
    model_name: str | None = None,
) -> Path:
    """Build the diagnostic figure and save it to *output_path*.

    Returns the resolved output path.
    """
    fig = make_figure(
        image_path,
        grid,
        ground_truth=ground_truth,
        title=title,
        tolerance=tolerance,
        model_name=model_name,
    )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    import matplotlib.pyplot as plt

    plt.close(fig)
    return output_path


# ---------------------------------------------------------------------------
# Three-panel comparison figure: image | predicted | ground truth
# ---------------------------------------------------------------------------


def make_comparison_figure(
    image_path: Path,
    predicted: "DailyRainfallGrid | None",
    ground_truth: "DailyRainfallGrid",
    title: str | None = None,
    tolerance: float = 0.005,
    model_name: str | None = None,
) -> "Figure":
    """Return a three-panel matplotlib Figure for validation comparisons.

    Panels (left to right):
      1. Source document image.
      2. Predicted grid — cells coloured blue (match within *tolerance*) or
         red (mismatch / missing).  Shows ``"[parse failed]"`` when *predicted*
         is ``None``.
      3. Ground-truth grid — all cells shown in black.

    Parameters
    ----------
    image_path:
        Path to the document image.
    predicted:
        The model-extracted rainfall grid, or ``None`` if parsing failed.
    ground_truth:
        The reference (transcription) rainfall grid.
    title:
        Figure suptitle.  Defaults to the image filename stem.
    tolerance:
        Absolute tolerance in inches for the blue/red colouring.
    model_name:
        Name of the model used for extraction.  Shown as the predicted-panel
        title.  Defaults to ``"Predicted"`` when ``None``.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image as PILImage

    img = PILImage.open(image_path).convert("RGB")
    img_w, img_h = img.size
    img_aspect = img_w / img_h

    # Each panel height; panel width follows image aspect ratio.
    panel_h = 11.0
    panel_w = panel_h * img_aspect
    title_margin = 0.4
    h_gap = 0.6
    l_margin = 0.2
    r_margin = 0.2
    fig_w = l_margin + panel_w + h_gap + panel_w + h_gap + panel_w + r_margin
    fig_h = panel_h + title_margin

    fig = plt.figure(figsize=(fig_w, fig_h))
    title = title or Path(image_path).stem
    fig.suptitle(title, fontsize=18, y=0.99)

    bottom = 0.01
    ax_h = panel_h / fig_h - 0.02
    panel_w_frac = panel_w / fig_w
    h_gap_frac = h_gap / fig_w

    img_left = l_margin / fig_w
    pred_left = img_left + panel_w_frac + h_gap_frac
    gt_left = pred_left + panel_w_frac + h_gap_frac

    # Table panels are slightly shorter (same adjustment as make_figure)
    tbl_ax_h = ax_h * 0.82
    tbl_bottom = bottom + tbl_ax_h * 0.07
    tbl_ax_h = tbl_ax_h * 0.93

    # ── Left panel: source image ──────────────────────────────────────────
    ax_img = fig.add_axes([img_left, bottom, panel_w_frac, ax_h])
    ax_img.imshow(img)
    ax_img.axis("off")

    # ── Middle panel: predicted grid ──────────────────────────────────────
    ax_pred = fig.add_axes([pred_left, tbl_bottom, panel_w_frac, tbl_ax_h])
    ax_pred.axis("off")
    pred_title = model_name if model_name is not None else "Predicted"
    ax_pred.set_title(pred_title, fontsize=16)

    if predicted is not None:
        pred_text = _cell_text(predicted)
        text_colours = _diff_text_colours(predicted, ground_truth, tolerance)
    else:
        # Parse failed: show empty cells with a header note.
        pred_text = [[""] * 12] * 32
        text_colours = [["red"] * 12] * 32
        ax_pred.text(
            0.5,
            0.5,
            "[parse failed]",
            ha="center",
            va="center",
            fontsize=14,
            color="red",
            transform=ax_pred.transAxes,
        )

    tbl_pred = ax_pred.table(
        cellText=pred_text,
        rowLabels=_ROW_LABELS,
        colLabels=_MONTH_ABBREVS,
        cellColours=[["white"] * 12] * 32,
        cellLoc="center",
        loc="center",
        bbox=[0, 0, 1, 1],
    )
    tbl_pred.auto_set_font_size(False)
    tbl_pred.set_fontsize(7)
    for (row, col), cell in tbl_pred.get_celld().items():
        cell.set_edgecolor("#aaaaaa")
        if row > 0 and col >= 0:
            cell.set_fontsize(10.5)
            cell.get_text().set_color(text_colours[row - 1][col])

    # ── Right panel: ground-truth grid ────────────────────────────────────
    ax_gt = fig.add_axes([gt_left, tbl_bottom, panel_w_frac, tbl_ax_h])
    ax_gt.axis("off")
    ax_gt.set_title("Ground truth", fontsize=16)

    gt_text = _cell_text(ground_truth)
    tbl_gt = ax_gt.table(
        cellText=gt_text,
        rowLabels=_ROW_LABELS,
        colLabels=_MONTH_ABBREVS,
        cellColours=[["white"] * 12] * 32,
        cellLoc="center",
        loc="center",
        bbox=[0, 0, 1, 1],
    )
    tbl_gt.auto_set_font_size(False)
    tbl_gt.set_fontsize(7)
    for (row, col), cell in tbl_gt.get_celld().items():
        cell.set_edgecolor("#aaaaaa")
        if row > 0 and col >= 0:
            cell.set_fontsize(10.5)
            cell.get_text().set_color("black")

    return fig


def save_comparison_figure(
    image_path: Path,
    predicted: "DailyRainfallGrid | None",
    ground_truth: "DailyRainfallGrid",
    output_path: Path,
    title: str | None = None,
    tolerance: float = 0.005,
    dpi: int = 150,
    model_name: str | None = None,
) -> Path:
    """Build the three-panel comparison figure and save it to *output_path*.

    Returns the resolved output path.
    """
    fig = make_comparison_figure(
        image_path,
        predicted,
        ground_truth,
        title=title,
        tolerance=tolerance,
        model_name=model_name,
    )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    import matplotlib.pyplot as plt

    plt.close(fig)
    return output_path
