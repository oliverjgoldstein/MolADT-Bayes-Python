from __future__ import annotations

import math
from html import escape
from pathlib import Path

import pandas as pd

from .common import ensure_directory

SPLIT_ORDER = ("train", "valid", "test")
SPLIT_COLORS = {
    "train": "#2563eb",
    "valid": "#f59e0b",
    "test": "#dc2626",
}
BACKGROUND = "#fcfbf7"
CARD_FILL = "#ffffff"
CARD_STROKE = "#d6d3d1"
TEXT = "#1f2937"
MUTED = "#6b7280"
GRID = "#d1d5db"
DIAGONAL = "#64748b"


def write_split_rmse_overview(metrics: pd.DataFrame, destination: Path) -> None:
    panels = list(_iter_metric_panels(metrics))
    if not panels:
        return
    cols = 2 if len(panels) > 1 else 1
    card_width = 420
    card_height = 250
    gap = 24
    margin = 24
    rows = math.ceil(len(panels) / cols)
    total_width = margin * 2 + cols * card_width + (cols - 1) * gap
    total_height = margin * 2 + rows * card_height + (rows - 1) * gap
    global_max_rmse = max(max(panel["rmse_by_split"].values()) for panel in panels)
    global_max_rmse = max(global_max_rmse, 1e-6) * 1.15

    parts = [_svg_header(total_width, total_height)]
    parts.append(f'<rect width="{total_width}" height="{total_height}" fill="{BACKGROUND}" />')
    for index, panel in enumerate(panels):
        row = index // cols
        col = index % cols
        x0 = margin + col * (card_width + gap)
        y0 = margin + row * (card_height + gap)
        parts.extend(_metric_card_svg(panel, x0=x0, y0=y0, width=card_width, height=card_height, y_max=global_max_rmse))
    parts.append("</svg>\n")
    ensure_directory(destination.parent)
    destination.write_text("".join(parts), encoding="utf-8")


def write_predicted_vs_actual_overview(predictions: pd.DataFrame, destination: Path, *, max_points_per_split: int = 500) -> None:
    panels = list(_iter_prediction_panels(predictions, max_points_per_split=max_points_per_split))
    if not panels:
        return
    cols = 2 if len(panels) > 1 else 1
    card_width = 420
    card_height = 300
    gap = 24
    margin = 24
    rows = math.ceil(len(panels) / cols)
    total_width = margin * 2 + cols * card_width + (cols - 1) * gap
    total_height = margin * 2 + rows * card_height + (rows - 1) * gap

    parts = [_svg_header(total_width, total_height)]
    parts.append(f'<rect width="{total_width}" height="{total_height}" fill="{BACKGROUND}" />')
    for index, panel in enumerate(panels):
        row = index // cols
        col = index % cols
        x0 = margin + col * (card_width + gap)
        y0 = margin + row * (card_height + gap)
        parts.extend(_prediction_card_svg(panel, x0=x0, y0=y0, width=card_width, height=card_height))
    parts.append("</svg>\n")
    ensure_directory(destination.parent)
    destination.write_text("".join(parts), encoding="utf-8")


def _iter_metric_panels(metrics: pd.DataFrame):
    order = ["dataset", "representation", "model", "method"]
    grouped = metrics.sort_values(order + ["split"]).groupby(order, sort=True)
    for (dataset, representation, model, method), frame in grouped:
        rmse_by_split = {
            split: float(frame.loc[frame["split"] == split, "rmse"].iloc[0])
            for split in SPLIT_ORDER
            if not frame.loc[frame["split"] == split].empty
        }
        mae_by_split = {
            split: float(frame.loc[frame["split"] == split, "mae"].iloc[0])
            for split in SPLIT_ORDER
            if not frame.loc[frame["split"] == split].empty
        }
        r2_by_split = {
            split: float(frame.loc[frame["split"] == split, "r2"].iloc[0])
            for split in SPLIT_ORDER
            if not frame.loc[frame["split"] == split].empty
        }
        if "train" not in rmse_by_split or "test" not in rmse_by_split:
            continue
        yield {
            "dataset": str(dataset),
            "representation": str(representation),
            "model": str(model),
            "method": str(method),
            "rmse_by_split": rmse_by_split,
            "mae_by_split": mae_by_split,
            "r2_by_split": r2_by_split,
        }


def _metric_card_svg(panel: dict[str, object], *, x0: int, y0: int, width: int, height: int, y_max: float) -> list[str]:
    plot_x = x0 + 44
    plot_y = y0 + 76
    plot_width = width - 68
    plot_height = 116
    bar_width = 52
    bar_gap = 30
    left_pad = 26
    title = f"{panel['dataset']} / {panel['representation']}"
    subtitle = f"{panel['model']} / {panel['method']}"
    rmse_by_split = panel["rmse_by_split"]
    mae_by_split = panel["mae_by_split"]
    r2_by_split = panel["r2_by_split"]
    train_rmse = float(rmse_by_split["train"])
    test_rmse = float(rmse_by_split["test"])
    gap = test_rmse - train_rmse
    parts = [
        f'<rect x="{x0}" y="{y0}" width="{width}" height="{height}" rx="18" fill="{CARD_FILL}" stroke="{CARD_STROKE}" />',
        f'<text x="{x0 + 20}" y="{y0 + 28}" font-size="18" font-family="Georgia, serif" fill="{TEXT}">{escape(title)}</text>',
        f'<text x="{x0 + 20}" y="{y0 + 50}" font-size="12" font-family="Helvetica, Arial, sans-serif" fill="{MUTED}">{escape(subtitle)}</text>',
    ]
    for step in range(5):
        y_value = y_max * step / 4.0
        y = plot_y + plot_height - (plot_height * step / 4.0)
        parts.append(f'<line x1="{plot_x}" y1="{y:.1f}" x2="{plot_x + plot_width}" y2="{y:.1f}" stroke="{GRID}" stroke-width="1" />')
        parts.append(
            f'<text x="{plot_x - 8}" y="{y + 4:.1f}" text-anchor="end" font-size="11" font-family="Helvetica, Arial, sans-serif" fill="{MUTED}">{y_value:.2f}</text>'
        )
    parts.append(f'<line x1="{plot_x}" y1="{plot_y + plot_height}" x2="{plot_x + plot_width}" y2="{plot_y + plot_height}" stroke="{TEXT}" stroke-width="1.5" />')
    for index, split in enumerate(SPLIT_ORDER):
        value = float(rmse_by_split.get(split, 0.0))
        bar_height = 0.0 if y_max <= 0 else (value / y_max) * plot_height
        x = plot_x + left_pad + index * (bar_width + bar_gap)
        y = plot_y + plot_height - bar_height
        parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_width}" height="{bar_height:.1f}" fill="{SPLIT_COLORS[split]}" opacity="0.9" />')
        parts.append(
            f'<text x="{x + bar_width / 2:.1f}" y="{plot_y + plot_height + 18}" text-anchor="middle" font-size="12" font-family="Helvetica, Arial, sans-serif" fill="{TEXT}">{escape(split)}</text>'
        )
        parts.append(
            f'<text x="{x + bar_width / 2:.1f}" y="{y - 8:.1f}" text-anchor="middle" font-size="11" font-family="Helvetica, Arial, sans-serif" fill="{TEXT}">{value:.3f}</text>'
        )
    parts.extend(
        [
            f'<text x="{x0 + 20}" y="{y0 + 214}" font-size="12" font-family="Helvetica, Arial, sans-serif" fill="{TEXT}">RMSE gap (test - train): {gap:+.3f}</text>',
            f'<text x="{x0 + 20}" y="{y0 + 232}" font-size="12" font-family="Helvetica, Arial, sans-serif" fill="{MUTED}">train MAE {float(mae_by_split["train"]):.3f}, test MAE {float(mae_by_split["test"]):.3f}, train R2 {float(r2_by_split["train"]):.3f}, test R2 {float(r2_by_split["test"]):.3f}</text>',
        ]
    )
    return parts


def _iter_prediction_panels(predictions: pd.DataFrame, *, max_points_per_split: int):
    order = ["dataset", "representation", "model", "method"]
    grouped = predictions.sort_values(order + ["split"]).groupby(order, sort=True)
    for (dataset, representation, model, method), frame in grouped:
        sampled_frames = []
        for split in SPLIT_ORDER:
            split_frame = frame.loc[frame["split"] == split, ["actual", "predicted_mean", "split"]].copy()
            if split_frame.empty:
                continue
            if len(split_frame) > max_points_per_split:
                split_frame = split_frame.sample(n=max_points_per_split, random_state=0).sort_index()
            sampled_frames.append(split_frame)
        if not sampled_frames:
            continue
        sampled = pd.concat(sampled_frames, ignore_index=True)
        minimum = float(min(sampled["actual"].min(), sampled["predicted_mean"].min()))
        maximum = float(max(sampled["actual"].max(), sampled["predicted_mean"].max()))
        if math.isclose(minimum, maximum):
            minimum -= 1.0
            maximum += 1.0
        padding = (maximum - minimum) * 0.08
        yield {
            "dataset": str(dataset),
            "representation": str(representation),
            "model": str(model),
            "method": str(method),
            "points": sampled,
            "minimum": minimum - padding,
            "maximum": maximum + padding,
        }


def _prediction_card_svg(panel: dict[str, object], *, x0: int, y0: int, width: int, height: int) -> list[str]:
    plot_x = x0 + 48
    plot_y = y0 + 74
    plot_width = width - 84
    plot_height = 170
    minimum = float(panel["minimum"])
    maximum = float(panel["maximum"])
    title = f"{panel['dataset']} / {panel['representation']}"
    subtitle = f"{panel['model']} / {panel['method']}"
    points: pd.DataFrame = panel["points"]
    parts = [
        f'<rect x="{x0}" y="{y0}" width="{width}" height="{height}" rx="18" fill="{CARD_FILL}" stroke="{CARD_STROKE}" />',
        f'<text x="{x0 + 20}" y="{y0 + 28}" font-size="18" font-family="Georgia, serif" fill="{TEXT}">{escape(title)}</text>',
        f'<text x="{x0 + 20}" y="{y0 + 50}" font-size="12" font-family="Helvetica, Arial, sans-serif" fill="{MUTED}">{escape(subtitle)}</text>',
    ]
    for step in range(5):
        fraction = step / 4.0
        x = plot_x + plot_width * fraction
        y = plot_y + plot_height - plot_height * fraction
        tick = minimum + (maximum - minimum) * fraction
        parts.append(f'<line x1="{plot_x}" y1="{plot_y + plot_height - plot_height * fraction:.1f}" x2="{plot_x + plot_width}" y2="{plot_y + plot_height - plot_height * fraction:.1f}" stroke="{GRID}" stroke-width="1" />')
        parts.append(f'<line x1="{x:.1f}" y1="{plot_y}" x2="{x:.1f}" y2="{plot_y + plot_height}" stroke="{GRID}" stroke-width="1" />')
        parts.append(
            f'<text x="{x:.1f}" y="{plot_y + plot_height + 18}" text-anchor="middle" font-size="11" font-family="Helvetica, Arial, sans-serif" fill="{MUTED}">{tick:.2f}</text>'
        )
        parts.append(
            f'<text x="{plot_x - 10}" y="{plot_y + plot_height - plot_height * fraction + 4:.1f}" text-anchor="end" font-size="11" font-family="Helvetica, Arial, sans-serif" fill="{MUTED}">{tick:.2f}</text>'
        )
    parts.append(f'<line x1="{plot_x}" y1="{plot_y + plot_height}" x2="{plot_x + plot_width}" y2="{plot_y}" stroke="{DIAGONAL}" stroke-width="1.5" stroke-dasharray="4 4" />')
    for _, row in points.iterrows():
        actual = float(row["actual"])
        predicted = float(row["predicted_mean"])
        split = str(row["split"])
        x = _scale(actual, minimum, maximum, plot_x, plot_x + plot_width)
        y = _scale(predicted, minimum, maximum, plot_y + plot_height, plot_y)
        parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="2.6" fill="{SPLIT_COLORS.get(split, TEXT)}" opacity="0.55" />')
    legend_x = x0 + 20
    legend_y = y0 + height - 28
    for index, split in enumerate(SPLIT_ORDER):
        x = legend_x + index * 106
        parts.append(f'<rect x="{x}" y="{legend_y - 10}" width="12" height="12" fill="{SPLIT_COLORS[split]}" />')
        parts.append(
            f'<text x="{x + 18}" y="{legend_y}" font-size="12" font-family="Helvetica, Arial, sans-serif" fill="{TEXT}">{escape(split)}</text>'
        )
    return parts


def _scale(value: float, source_min: float, source_max: float, target_min: float, target_max: float) -> float:
    if math.isclose(source_min, source_max):
        return (target_min + target_max) / 2.0
    ratio = (value - source_min) / (source_max - source_min)
    return target_min + ratio * (target_max - target_min)


def _svg_header(width: int, height: int) -> str:
    return f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-label="Predictive benchmark overview">'
