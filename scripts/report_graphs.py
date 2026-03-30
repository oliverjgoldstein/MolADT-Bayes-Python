from __future__ import annotations

import math
from html import escape
from pathlib import Path

import pandas as pd

from .common import ensure_directory

SPLIT_COLORS = {
    "train": "#2563eb",
    "test": "#dc2626",
}
BACKGROUND = "#fcfbf7"
CARD_FILL = "#ffffff"
CARD_STROKE = "#d6d3d1"
TEXT = "#1f2937"
MUTED = "#6b7280"
GRID = "#d1d5db"
LITERATURE = "#374151"
TIMING = "#0f766e"


def write_review_rmse_overview(review_frame: pd.DataFrame, destination: Path) -> None:
    if review_frame.empty:
        return
    rows = review_frame.sort_values(["dataset", "representation", "task"]).reset_index(drop=True)
    cols = 2 if len(rows) > 1 else 1
    card_width = 430
    card_height = 262
    gap = 24
    margin = 24
    panel_rows = math.ceil(len(rows) / cols)
    total_width = margin * 2 + cols * card_width + (cols - 1) * gap
    total_height = margin * 2 + panel_rows * card_height + (panel_rows - 1) * gap
    literature_values = rows["literature_rmse"].dropna().astype(float)
    y_max = max(
        float(rows["train_rmse"].max()),
        float(rows["test_rmse"].max()),
        float(literature_values.max()) if not literature_values.empty else 0.0,
        1e-6,
    ) * 1.15

    parts = [_svg_header(total_width, total_height)]
    parts.append(f'<rect width="{total_width}" height="{total_height}" fill="{BACKGROUND}" />')
    for index, (_, row) in enumerate(rows.iterrows()):
        grid_row = index // cols
        grid_col = index % cols
        x0 = margin + grid_col * (card_width + gap)
        y0 = margin + grid_row * (card_height + gap)
        parts.extend(_review_rmse_card_svg(row, x0=x0, y0=y0, width=card_width, height=card_height, y_max=y_max))
    parts.append("</svg>\n")
    ensure_directory(destination.parent)
    destination.write_text("".join(parts), encoding="utf-8")


def write_timing_stage_overview(timing: pd.DataFrame, destination: Path) -> None:
    if timing.empty:
        return
    stage_order = ["raw_file_read", "smiles_parse_sanitize", "smiles_canonicalization", "moladt_parse_render"]
    order_index = {stage: index for index, stage in enumerate(stage_order)}
    rows = timing.copy()
    rows["_order"] = rows["stage"].map(order_index).fillna(len(stage_order)).astype(int)
    rows = rows.sort_values(["_order", "stage"]).reset_index(drop=True)
    card_width = 920
    card_height = 130 + len(rows) * 58
    x0 = 24
    y0 = 24
    total_width = card_width + 48
    total_height = card_height + 48
    chart_x = x0 + 220
    chart_width = 620
    max_rate = max(float(rows["molecules_per_second"].max()), 1e-6) * 1.1

    parts = [_svg_header(total_width, total_height)]
    parts.append(f'<rect width="{total_width}" height="{total_height}" fill="{BACKGROUND}" />')
    parts.append(f'<rect x="{x0}" y="{y0}" width="{card_width}" height="{card_height}" rx="18" fill="{CARD_FILL}" stroke="{CARD_STROKE}" />')
    parts.append(f'<text x="{x0 + 24}" y="{y0 + 32}" font-size="20" font-family="Georgia, serif" fill="{TEXT}">Local timing overview</text>')
    parts.append(
        f'<text x="{x0 + 24}" y="{y0 + 54}" font-size="12" font-family="Helvetica, Arial, sans-serif" fill="{MUTED}">Molecules per second by stage, with runtime and success counts shown alongside.</text>'
    )
    for _, row in rows.iterrows():
        stage = str(row["stage"])
        stage_index = int(row["_order"])
        y = y0 + 86 + stage_index * 58
        bar_width = 0.0 if max_rate <= 0 else chart_width * (float(row["molecules_per_second"]) / max_rate)
        parts.append(f'<text x="{x0 + 24}" y="{y + 16}" font-size="13" font-family="Helvetica, Arial, sans-serif" fill="{TEXT}">{escape(stage)}</text>')
        parts.append(f'<rect x="{chart_x}" y="{y}" width="{chart_width}" height="20" rx="10" fill="#e7e5e4" />')
        parts.append(f'<rect x="{chart_x}" y="{y}" width="{bar_width:.1f}" height="20" rx="10" fill="{TIMING}" />')
        parts.append(
            f'<text x="{chart_x + chart_width + 12}" y="{y + 15}" font-size="12" font-family="Helvetica, Arial, sans-serif" fill="{TEXT}">{float(row["molecules_per_second"]):.1f} mol/s</text>'
        )
        parts.append(
            f'<text x="{chart_x}" y="{y + 40}" font-size="11" font-family="Helvetica, Arial, sans-serif" fill="{MUTED}">runtime {float(row["total_runtime_seconds"]):.3f}s; success {int(row["success_count"])}/{int(row["molecule_count"])}</text>'
        )
    parts.append("</svg>\n")
    ensure_directory(destination.parent)
    destination.write_text("".join(parts), encoding="utf-8")


def _review_rmse_card_svg(row: pd.Series, *, x0: int, y0: int, width: int, height: int, y_max: float) -> list[str]:
    plot_x = x0 + 44
    plot_y = y0 + 76
    plot_width = width - 68
    plot_height = 116
    title = str(row["task"])
    subtitle = f"{row['model']} / {row['method']}"
    bars = [
        ("train", float(row["train_rmse"]), SPLIT_COLORS["train"]),
        ("test", float(row["test_rmse"]), SPLIT_COLORS["test"]),
    ]
    literature_value = row["literature_rmse"]
    if pd.notna(literature_value):
        bars.append(("paper", float(literature_value), LITERATURE))
    bar_width = 52
    bar_gap = 22
    left_pad = 16
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
    for index, (label, value, color) in enumerate(bars):
        bar_height = 0.0 if y_max <= 0 else (value / y_max) * plot_height
        x = plot_x + left_pad + index * (bar_width + bar_gap)
        y = plot_y + plot_height - bar_height
        parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_width}" height="{bar_height:.1f}" fill="{color}" opacity="0.9" />')
        parts.append(
            f'<text x="{x + bar_width / 2:.1f}" y="{plot_y + plot_height + 18}" text-anchor="middle" font-size="12" font-family="Helvetica, Arial, sans-serif" fill="{TEXT}">{escape(label)}</text>'
        )
        parts.append(
            f'<text x="{x + bar_width / 2:.1f}" y="{y - 8:.1f}" text-anchor="middle" font-size="11" font-family="Helvetica, Arial, sans-serif" fill="{TEXT}">{value:.3f}</text>'
        )
    parts.extend(
        [
            f'<text x="{x0 + 20}" y="{y0 + 218}" font-size="12" font-family="Helvetica, Arial, sans-serif" fill="{TEXT}">RMSE gap (test - train): {float(row["test_minus_train_rmse"]):+.3f}</text>',
            f'<text x="{x0 + 20}" y="{y0 + 236}" font-size="11" font-family="Helvetica, Arial, sans-serif" fill="{MUTED}">{escape(str(row["literature_display"]))}</text>',
            f'<text x="{x0 + 20}" y="{y0 + 252}" font-size="10" font-family="Helvetica, Arial, sans-serif" fill="{MUTED}">{escape(str(row["note"]))}</text>',
        ]
    )
    return parts


def _svg_header(width: int, height: int) -> str:
    return f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-label="Benchmark overview">'
