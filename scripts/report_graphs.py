from __future__ import annotations

import math
from html import escape
from pathlib import Path
from textwrap import wrap

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
SERIES_COLORS = {
    "smiles": "#b45309",
    "moladt": "#0f766e",
    "sdf_geom": "#0369a1",
    "moladt_geom": "#be123c",
    "paper": LITERATURE,
}


def write_moleculenet_comparison_overviews(comparison_frame: pd.DataFrame, destination_dir: Path) -> None:
    if comparison_frame.empty:
        return
    ensure_directory(destination_dir)
    for _, row in comparison_frame.sort_values(["dataset"]).iterrows():
        dataset = str(row["dataset"])
        metric_name = str(row["metric_name"])
        destination = destination_dir / f"{dataset}_{metric_name.lower()}_vs_moleculenet.svg"
        train_value = float(row.get("train_value", row["local_value"]))
        test_value = float(row.get("test_value", row["local_value"]))
        paper_value = float(row["paper_value"])
        selection_split = str(row.get("selection_split", "valid"))
        selection_label = "Validation" if selection_split == "valid" else selection_split.title()
        value_max = max(train_value, test_value, paper_value, 1e-6) * 1.2
        width = 640
        height = 332
        x0 = 24
        y0 = 24
        plot_x = x0 + 56
        plot_y = y0 + 104
        plot_width = width - 112
        plot_height = 124
        bar_width = 92
        bar_gap = 48
        bars = [
            ("Training", train_value, SPLIT_COLORS["train"]),
            ("Test", test_value, SPLIT_COLORS["test"]),
            ("Paper", paper_value, SERIES_COLORS["paper"]),
        ]
        parts = [_svg_header(width, height)]
        parts.append(f'<rect width="{width}" height="{height}" fill="{BACKGROUND}" />')
        parts.append(f'<rect x="{x0}" y="{y0}" width="{width - 48}" height="{height - 48}" rx="18" fill="{CARD_FILL}" stroke="{CARD_STROKE}" />')
        parts.append(
            f'<text x="{x0 + 20}" y="{y0 + 30}" font-size="22" font-family="Georgia, serif" fill="{TEXT}">'
            f"{escape(str(row['dataset_label']))}: {escape(metric_name)}"
            "</text>"
        )
        parts.append(
            f'<text x="{x0 + 20}" y="{y0 + 50}" font-size="12" font-family="Helvetica, Arial, sans-serif" fill="{MUTED}">'
            "Training and held-out test scores from the validation-selected local Stan MolADT run, plus the cited MoleculeNet Table 3 baseline."
            "</text>"
        )
        parts.append(
            f'<text x="{x0 + 20}" y="{y0 + 68}" font-size="11" font-family="Helvetica, Arial, sans-serif" fill="{MUTED}">'
            f"Local: {escape(str(row['model']))} / {escape(str(row['method']))}"
            "</text>"
        )
        parts.append(
            f'<text x="{x0 + 20}" y="{y0 + 84}" font-size="11" font-family="Helvetica, Arial, sans-serif" fill="{MUTED}">'
            f"Selection: {escape(selection_label)} {escape(metric_name)}; Paper: {escape(str(row['paper_model_name']))} from MoleculeNet"
            "</text>"
        )
        for step in range(5):
            fraction = step / 4.0
            y_value = value_max * fraction
            y = plot_y + plot_height - plot_height * fraction
            parts.append(f'<line x1="{plot_x}" y1="{y:.1f}" x2="{plot_x + plot_width}" y2="{y:.1f}" stroke="{GRID}" stroke-width="1" />')
            parts.append(
                f'<text x="{plot_x - 8}" y="{y + 4:.1f}" text-anchor="end" font-size="10" font-family="Helvetica, Arial, sans-serif" fill="{MUTED}">{y_value:.2f}</text>'
            )
        parts.append(f'<line x1="{plot_x}" y1="{plot_y + plot_height}" x2="{plot_x + plot_width}" y2="{plot_y + plot_height}" stroke="{TEXT}" stroke-width="1.4" />')
        total_bar_width = len(bars) * bar_width + (len(bars) - 1) * bar_gap
        left_pad = plot_x + (plot_width - total_bar_width) / 2
        for index, (label, value, color) in enumerate(bars):
            x = left_pad + index * (bar_width + bar_gap)
            bar_height = (value / value_max) * plot_height if value_max > 0 else 0.0
            y = plot_y + plot_height - bar_height
            parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_width}" height="{bar_height:.1f}" rx="10" fill="{color}" opacity="0.94" />')
            parts.append(
                f'<text x="{x + bar_width / 2:.1f}" y="{y - 8:.1f}" text-anchor="middle" font-size="12" font-family="Helvetica, Arial, sans-serif" fill="{TEXT}">{value:.3f}</text>'
            )
            parts.append(
                f'<text x="{x + bar_width / 2:.1f}" y="{plot_y + plot_height + 22}" text-anchor="middle" font-size="12" font-family="Helvetica, Arial, sans-serif" fill="{TEXT}">{escape(label)}</text>'
            )
        note_lines = _wrap_text(str(row.get("note", "")), 68)[:3]
        for line_index, line in enumerate(note_lines):
            parts.append(
                f'<text x="{x0 + 20}" y="{height - 54 + line_index * 13}" font-size="10" font-family="Helvetica, Arial, sans-serif" fill="{MUTED}">{escape(line)}</text>'
            )
        parts.append("</svg>\n")
        destination.write_text("".join(parts), encoding="utf-8")


def write_review_rmse_overview(review_frame: pd.DataFrame, destination: Path) -> None:
    if review_frame.empty:
        return
    rows = review_frame.sort_values(["dataset", "representation", "task"]).reset_index(drop=True)
    cols = 2 if len(rows) > 1 else 1
    card_width = 430
    card_height = 302
    gap = 24
    margin = 24
    header_height = 62
    panel_rows = math.ceil(len(rows) / cols)
    total_width = margin * 2 + cols * card_width + (cols - 1) * gap
    total_height = margin * 2 + header_height + panel_rows * card_height + (panel_rows - 1) * gap
    literature_values = rows["literature_rmse"].dropna().astype(float)
    y_max = max(
        float(rows["train_rmse"].max()),
        float(rows["test_rmse"].max()),
        float(literature_values.max()) if not literature_values.empty else 0.0,
        1e-6,
    ) * 1.15

    parts = [_svg_header(total_width, total_height)]
    parts.append(f'<rect width="{total_width}" height="{total_height}" fill="{BACKGROUND}" />')
    parts.append(f'<text x="{margin}" y="{margin + 22}" font-size="22" font-family="Georgia, serif" fill="{TEXT}">Best local train/test RMSE against paper context</text>')
    parts.append(
        f'<text x="{margin}" y="{margin + 42}" font-size="12" font-family="Helvetica, Arial, sans-serif" fill="{MUTED}">'
        "Each card shows the best descriptor-based Bayesian run for one dataset/representation. "
        "Blue is local train RMSE, red is local test RMSE, and gray is the cited external neural baseline when available."
        "</text>"
    )
    for index, (_, row) in enumerate(rows.iterrows()):
        grid_row = index // cols
        grid_col = index % cols
        x0 = margin + grid_col * (card_width + gap)
        y0 = margin + header_height + grid_row * (card_height + gap)
        parts.extend(_review_rmse_card_svg(row, x0=x0, y0=y0, width=card_width, height=card_height, y_max=y_max))
    parts.append("</svg>\n")
    ensure_directory(destination.parent)
    destination.write_text("".join(parts), encoding="utf-8")


def write_inference_sweep_overview(metrics: pd.DataFrame, destination: Path) -> None:
    if metrics.empty:
        return
    test_rows = metrics.loc[metrics["split"] == "test"].copy()
    if test_rows.empty:
        return
    groups = [
        (key, frame.sort_values(["rmse", "mae", "runtime_seconds", "model", "method"]).reset_index(drop=True))
        for key, frame in test_rows.groupby(["dataset", "representation"], sort=True)
    ]
    card_width = 460
    gap = 24
    margin = 24
    header_height = 66
    cols = 2 if len(groups) > 1 else 1
    max_rows = max(len(frame) for _, frame in groups)
    card_height = 96 + max_rows * 34
    panel_rows = math.ceil(len(groups) / cols)
    total_width = margin * 2 + cols * card_width + (cols - 1) * gap
    total_height = margin * 2 + header_height + panel_rows * card_height + (panel_rows - 1) * gap
    rmse_max = max(float(test_rows["rmse"].max()), 1e-6) * 1.1

    parts = [_svg_header(total_width, total_height)]
    parts.append(f'<rect width="{total_width}" height="{total_height}" fill="{BACKGROUND}" />')
    parts.append(f'<text x="{margin}" y="{margin + 22}" font-size="22" font-family="Georgia, serif" fill="{TEXT}">Inference sweep on local test splits</text>')
    parts.append(
        f'<text x="{margin}" y="{margin + 42}" font-size="12" font-family="Helvetica, Arial, sans-serif" fill="{MUTED}">'
        "Each bar is one fitted Stan model and inference-method combination on the test split. Lower RMSE is better."
        "</text>"
    )
    for index, ((dataset, representation), frame) in enumerate(groups):
        grid_row = index // cols
        grid_col = index % cols
        x0 = margin + grid_col * (card_width + gap)
        y0 = margin + header_height + grid_row * (card_height + gap)
        parts.extend(
            _inference_sweep_card_svg(
                frame,
                dataset=str(dataset),
                representation=str(representation),
                x0=x0,
                y0=y0,
                width=card_width,
                height=card_height,
                rmse_max=rmse_max,
            )
        )
    parts.append("</svg>\n")
    ensure_directory(destination.parent)
    destination.write_text("".join(parts), encoding="utf-8")


def write_timing_stage_overview(timing: pd.DataFrame, destination: Path) -> None:
    if timing.empty:
        return
    stage_order = [
        "raw_file_read",
        "smiles_parse_sanitize",
        "smiles_canonicalization",
        "timing_library_prepare",
        "smiles_csv_string_parse",
        "smiles_library_parse",
        "moladt_file_parse",
        "moladt_parse_render",
    ]
    order_index = {stage: index for index, stage in enumerate(stage_order)}
    rows = timing.copy()
    rows["_order"] = rows["stage"].map(order_index).fillna(len(stage_order)).astype(int)
    rows = rows.sort_values(["_order", "stage"]).reset_index(drop=True)
    card_width = 980
    card_height = 144 + len(rows) * 78
    x0 = 24
    y0 = 24
    total_width = card_width + 48
    total_height = card_height + 48
    chart_x = x0 + 318
    chart_width = 510
    max_rate = max(float(rows["molecules_per_second"].max()), 1e-6) * 1.1

    parts = [_svg_header(total_width, total_height)]
    parts.append(f'<rect width="{total_width}" height="{total_height}" fill="{BACKGROUND}" />')
    parts.append(f'<rect x="{x0}" y="{y0}" width="{card_width}" height="{card_height}" rx="18" fill="{CARD_FILL}" stroke="{CARD_STROKE}" />')
    parts.append(f'<text x="{x0 + 24}" y="{y0 + 32}" font-size="20" font-family="Georgia, serif" fill="{TEXT}">Local timing overview</text>')
    parts.append(
        f'<text x="{x0 + 24}" y="{y0 + 54}" font-size="12" font-family="Helvetica, Arial, sans-serif" fill="{MUTED}">'
        "This is an execution pipeline view: raw file IO, optional toolkit-assisted SMILES normalization, and the local MolADT ingest/render stages."
        "</text>"
    )
    for _, row in rows.iterrows():
        stage = str(row["stage"])
        stage_index = int(row["_order"])
        y = y0 + 88 + stage_index * 78
        bar_width = 0.0 if max_rate <= 0 else chart_width * (float(row["molecules_per_second"]) / max_rate)
        description = str(row.get("description", ""))
        parts.append(f'<text x="{x0 + 24}" y="{y + 16}" font-size="13" font-family="Helvetica, Arial, sans-serif" fill="{TEXT}">{escape(stage)}</text>')
        for line_index, line in enumerate(_wrap_text(description, 38)):
            parts.append(
                f'<text x="{x0 + 24}" y="{y + 32 + line_index * 13}" font-size="11" font-family="Helvetica, Arial, sans-serif" fill="{MUTED}">{escape(line)}</text>'
            )
        parts.append(f'<rect x="{chart_x}" y="{y}" width="{chart_width}" height="20" rx="10" fill="#e7e5e4" />')
        parts.append(f'<rect x="{chart_x}" y="{y}" width="{bar_width:.1f}" height="20" rx="10" fill="{TIMING}" />')
        parts.append(
            f'<text x="{chart_x + chart_width + 12}" y="{y + 15}" font-size="12" font-family="Helvetica, Arial, sans-serif" fill="{TEXT}">{float(row["molecules_per_second"]):.1f} mol/s</text>'
        )
        parts.append(
            f'<text x="{chart_x}" y="{y + 42}" font-size="11" font-family="Helvetica, Arial, sans-serif" fill="{MUTED}">'
            f"runtime {float(row.get('total_runtime_seconds', 0.0)):.3f}s; "
            f"success {int(row.get('success_count', 0))}/{int(row.get('molecule_count', 0))}; "
            f"failures {int(row.get('failure_count', 0))}; median {float(row.get('median_latency_us', 0.0)):.1f} us"
            "</text>"
        )
    parts.append("</svg>\n")
    ensure_directory(destination.parent)
    destination.write_text("".join(parts), encoding="utf-8")


def write_predicted_vs_actual_overview(predictions: pd.DataFrame, destination: Path) -> None:
    if predictions.empty:
        return
    groups = [
        (key, frame.reset_index(drop=True))
        for key, frame in predictions.groupby(["dataset", "representation"], sort=True)
    ]
    cols = 2 if len(groups) > 1 else 1
    card_width = 430
    card_height = 320
    gap = 24
    margin = 24
    header_height = 66
    panel_rows = math.ceil(len(groups) / cols)
    total_width = margin * 2 + cols * card_width + (cols - 1) * gap
    total_height = margin * 2 + header_height + panel_rows * card_height + (panel_rows - 1) * gap
    all_values = pd.concat([predictions["actual"], predictions["predicted_mean"]], ignore_index=True).astype(float)
    value_min = float(all_values.min())
    value_max = float(all_values.max())
    if value_min == value_max:
        value_min -= 1.0
        value_max += 1.0

    parts = [_svg_header(total_width, total_height)]
    parts.append(f'<rect width="{total_width}" height="{total_height}" fill="{BACKGROUND}" />')
    parts.append(f'<text x="{margin}" y="{margin + 22}" font-size="22" font-family="Georgia, serif" fill="{TEXT}">Predicted vs actual on selected test runs</text>')
    parts.append(
        f'<text x="{margin}" y="{margin + 42}" font-size="12" font-family="Helvetica, Arial, sans-serif" fill="{MUTED}">'
        "Each dot is one molecule from the selected test split. The diagonal is perfect prediction."
        "</text>"
    )
    for index, ((dataset, representation), frame) in enumerate(groups):
        grid_row = index // cols
        grid_col = index % cols
        x0 = margin + grid_col * (card_width + gap)
        y0 = margin + header_height + grid_row * (card_height + gap)
        parts.extend(
            _scatter_card_svg(
                frame=frame,
                x0=x0,
                y0=y0,
                width=card_width,
                height=card_height,
                value_min=value_min,
                value_max=value_max,
                title=f"{dataset} / {representation}",
                subtitle="Dots are molecules; diagonal is ideal.",
                x_column="actual",
                y_column="predicted_mean",
            )
        )
    parts.append("</svg>\n")
    ensure_directory(destination.parent)
    destination.write_text("".join(parts), encoding="utf-8")


def write_residual_vs_uncertainty_overview(predictions: pd.DataFrame, destination: Path) -> None:
    if predictions.empty:
        return
    frame = predictions.copy()
    frame["absolute_residual"] = (frame["predicted_mean"].astype(float) - frame["actual"].astype(float)).abs()
    groups = [
        (key, group.reset_index(drop=True))
        for key, group in frame.groupby(["dataset", "representation"], sort=True)
    ]
    cols = 2 if len(groups) > 1 else 1
    card_width = 430
    card_height = 320
    gap = 24
    margin = 24
    header_height = 66
    panel_rows = math.ceil(len(groups) / cols)
    total_width = margin * 2 + cols * card_width + (cols - 1) * gap
    total_height = margin * 2 + header_height + panel_rows * card_height + (panel_rows - 1) * gap
    x_max = max(float(frame["predictive_sd"].astype(float).max()), 1e-6)
    y_max = max(float(frame["absolute_residual"].astype(float).max()), 1e-6)

    parts = [_svg_header(total_width, total_height)]
    parts.append(f'<rect width="{total_width}" height="{total_height}" fill="{BACKGROUND}" />')
    parts.append(f'<text x="{margin}" y="{margin + 22}" font-size="22" font-family="Georgia, serif" fill="{TEXT}">Residual vs uncertainty on selected test runs</text>')
    parts.append(
        f'<text x="{margin}" y="{margin + 42}" font-size="12" font-family="Helvetica, Arial, sans-serif" fill="{MUTED}">'
        "X is predictive standard deviation and Y is absolute error. Well-calibrated models should place larger errors on the right."
        "</text>"
    )
    for index, ((dataset, representation), group) in enumerate(groups):
        grid_row = index // cols
        grid_col = index % cols
        x0 = margin + grid_col * (card_width + gap)
        y0 = margin + header_height + grid_row * (card_height + gap)
        parts.extend(
            _scatter_card_svg(
                frame=group,
                x0=x0,
                y0=y0,
                width=card_width,
                height=card_height,
                value_min=0.0,
                value_max=max(x_max, y_max),
                title=f"{dataset} / {representation}",
                subtitle="X=predicted sd, Y=absolute residual.",
                x_column="predictive_sd",
                y_column="absolute_residual",
            )
        )
    parts.append("</svg>\n")
    ensure_directory(destination.parent)
    destination.write_text("".join(parts), encoding="utf-8")


def write_calibration_overview(calibration: pd.DataFrame, destination: Path) -> None:
    if calibration.empty:
        return
    test_rows = calibration.loc[calibration["split"] == "test"].copy()
    if test_rows.empty:
        return
    groups = [
        (key, frame.sort_values("nominal_coverage").reset_index(drop=True))
        for key, frame in test_rows.groupby(["dataset", "representation"], sort=True)
    ]
    cols = 2 if len(groups) > 1 else 1
    card_width = 430
    card_height = 300
    gap = 24
    margin = 24
    header_height = 66
    panel_rows = math.ceil(len(groups) / cols)
    total_width = margin * 2 + cols * card_width + (cols - 1) * gap
    total_height = margin * 2 + header_height + panel_rows * card_height + (panel_rows - 1) * gap

    parts = [_svg_header(total_width, total_height)]
    parts.append(f'<rect width="{total_width}" height="{total_height}" fill="{BACKGROUND}" />')
    parts.append(f'<text x="{margin}" y="{margin + 22}" font-size="22" font-family="Georgia, serif" fill="{TEXT}">Coverage calibration on selected test runs</text>')
    parts.append(
        f'<text x="{margin}" y="{margin + 42}" font-size="12" font-family="Helvetica, Arial, sans-serif" fill="{MUTED}">'
        "The dashed diagonal is perfect calibration. Points below it are overconfident and points above it are conservative."
        "</text>"
    )
    for index, ((dataset, representation), frame) in enumerate(groups):
        grid_row = index // cols
        grid_col = index % cols
        x0 = margin + grid_col * (card_width + gap)
        y0 = margin + header_height + grid_row * (card_height + gap)
        parts.extend(_calibration_card_svg(frame, x0=x0, y0=y0, width=card_width, height=card_height, title=f"{dataset} / {representation}"))
    parts.append("</svg>\n")
    ensure_directory(destination.parent)
    destination.write_text("".join(parts), encoding="utf-8")


def write_metric_comparison_overviews(comparison_frame: pd.DataFrame, destination_dir: Path) -> None:
    if comparison_frame.empty:
        return
    ensure_directory(destination_dir)
    rows = comparison_frame.copy()
    if "comparison_key" not in rows.columns:
        rows["comparison_key"] = "tabular"
    if "comparison_subtitle" not in rows.columns:
        rows["comparison_subtitle"] = rows["comparison_key"].map(_default_metric_comparison_subtitle)
    if "series_order" not in rows.columns:
        rows["series_order"] = rows["series_key"].map(lambda key: _series_order_for_key(str(key)))
    for (comparison_key, metric_key), metric_frame in rows.groupby(["comparison_key", "metric_key"], sort=False):
        rows = metric_frame.reset_index(drop=True)
        datasets = [
            (str(dataset), frame.reset_index(drop=True))
            for dataset, frame in rows.groupby("dataset", sort=True)
        ]
        if not datasets:
            continue
        cols = 2 if len(datasets) > 1 else 1
        card_width = 430
        card_height = 316
        gap = 24
        margin = 24
        header_height = 70
        panel_rows = math.ceil(len(datasets) / cols)
        total_width = margin * 2 + cols * card_width + (cols - 1) * gap
        total_height = margin * 2 + header_height + panel_rows * card_height + (panel_rows - 1) * gap
        value_min = float(rows["value"].min())
        value_max = float(rows["value"].max())
        if value_min == value_max:
            value_min -= 1.0
            value_max += 1.0
        y_min = min(0.0, value_min * 1.05)
        y_max = max(0.0, value_max * 1.15)
        if y_min == y_max:
            y_max = y_min + 1.0
        metric_label = str(rows.iloc[0]["metric_label"])
        subtitle = str(rows.iloc[0].get("comparison_subtitle", _default_metric_comparison_subtitle(str(comparison_key))))
        parts = [_svg_header(total_width, total_height)]
        parts.append(f'<rect width="{total_width}" height="{total_height}" fill="{BACKGROUND}" />')
        parts.append(
            f'<text x="{margin}" y="{margin + 22}" font-size="22" font-family="Georgia, serif" fill="{TEXT}">'
            f"{escape(_metric_comparison_title(metric_label, str(comparison_key)))}</text>"
        )
        parts.append(
            f'<text x="{margin}" y="{margin + 42}" font-size="12" font-family="Helvetica, Arial, sans-serif" fill="{MUTED}">'
            f"{escape(subtitle)}"
            "</text>"
        )
        for index, (dataset, frame) in enumerate(datasets):
            grid_row = index // cols
            grid_col = index % cols
            x0 = margin + grid_col * (card_width + gap)
            y0 = margin + header_height + grid_row * (card_height + gap)
            parts.extend(
                _metric_comparison_card_svg(
                    frame,
                    dataset=dataset,
                    x0=x0,
                    y0=y0,
                    width=card_width,
                    height=card_height,
                    y_min=y_min,
                    y_max=y_max,
                )
            )
        parts.append("</svg>\n")
        destination = destination_dir / _metric_comparison_filename(str(metric_key), str(comparison_key))
        destination.write_text("".join(parts), encoding="utf-8")


def _review_rmse_card_svg(row: pd.Series, *, x0: int, y0: int, width: int, height: int, y_max: float) -> list[str]:
    plot_x = x0 + 44
    plot_y = y0 + 86
    plot_width = width - 68
    plot_height = 110
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
        f'<text x="{x0 + 20}" y="{y0 + 66}" font-size="11" font-family="Helvetica, Arial, sans-serif" fill="{MUTED}">Blue=train local RMSE, red=test local RMSE, gray=paper context.</text>',
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
    parts.append(
        f'<text x="{x0 + 20}" y="{y0 + 222}" font-size="12" font-family="Helvetica, Arial, sans-serif" fill="{TEXT}">RMSE gap (test - train): {float(row["test_minus_train_rmse"]):+.3f}</text>'
    )
    literature_lines = _wrap_text(str(row["literature_display"]), 60)
    note_lines = _wrap_text(str(row["note"]), 64)
    for line_index, line in enumerate(literature_lines[:2]):
        parts.append(
            f'<text x="{x0 + 20}" y="{y0 + 240 + line_index * 14}" font-size="11" font-family="Helvetica, Arial, sans-serif" fill="{MUTED}">{escape(line)}</text>'
        )
    note_base_y = y0 + 240 + max(1, len(literature_lines[:2])) * 14
    for line_index, line in enumerate(note_lines[:3]):
        parts.append(
            f'<text x="{x0 + 20}" y="{note_base_y + line_index * 13}" font-size="10" font-family="Helvetica, Arial, sans-serif" fill="{MUTED}">{escape(line)}</text>'
        )
    return parts


def _inference_sweep_card_svg(
    frame: pd.DataFrame,
    *,
    dataset: str,
    representation: str,
    x0: int,
    y0: int,
    width: int,
    height: int,
    rmse_max: float,
) -> list[str]:
    left_label_x = x0 + 20
    chart_x = x0 + 208
    chart_width = width - 236
    split_scheme = str(frame.iloc[0].get("split_scheme", ""))
    train_count = int(frame.iloc[0]["n_train"])
    test_count = int(frame.iloc[0]["n_eval"])
    parts = [
        f'<rect x="{x0}" y="{y0}" width="{width}" height="{height}" rx="18" fill="{CARD_FILL}" stroke="{CARD_STROKE}" />',
        f'<text x="{x0 + 20}" y="{y0 + 28}" font-size="18" font-family="Georgia, serif" fill="{TEXT}">{escape(dataset)} / {escape(representation)}</text>',
        f'<text x="{x0 + 20}" y="{y0 + 48}" font-size="11" font-family="Helvetica, Arial, sans-serif" fill="{MUTED}">test rows sorted by RMSE; train={train_count}, test={test_count}, split={escape(split_scheme)}</text>',
    ]
    for row_index, (_, row) in enumerate(frame.iterrows()):
        y = y0 + 72 + row_index * 34
        bar_width = 0.0 if rmse_max <= 0 else chart_width * (float(row["rmse"]) / rmse_max)
        label = f"{row['model']} / {row['method']}"
        parts.append(f'<text x="{left_label_x}" y="{y + 12}" font-size="10" font-family="Helvetica, Arial, sans-serif" fill="{TEXT}">{escape(label)}</text>')
        parts.append(f'<rect x="{chart_x}" y="{y}" width="{chart_width}" height="14" rx="7" fill="#e7e5e4" />')
        parts.append(f'<rect x="{chart_x}" y="{y}" width="{bar_width:.1f}" height="14" rx="7" fill="#b45309" opacity="0.92" />')
        parts.append(
            f'<text x="{chart_x + chart_width + 8}" y="{y + 11}" font-size="10" font-family="Helvetica, Arial, sans-serif" fill="{TEXT}">{float(row["rmse"]):.3f}</text>'
        )
        parts.append(
            f'<text x="{chart_x}" y="{y + 28}" font-size="10" font-family="Helvetica, Arial, sans-serif" fill="{MUTED}">MAE {float(row["mae"]):.3f}; runtime {float(row["runtime_seconds"]):.2f}s; 90% coverage {float(row["coverage_90"]):.2f}</text>'
        )
    return parts


def _wrap_text(text: str, max_chars: int) -> list[str]:
    stripped = text.strip()
    if not stripped:
        return []
    return wrap(stripped, width=max_chars, break_long_words=False, break_on_hyphens=False)


def _svg_header(width: int, height: int) -> str:
    return f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-label="Benchmark overview">'


def _scatter_card_svg(
    *,
    frame: pd.DataFrame,
    x0: int,
    y0: int,
    width: int,
    height: int,
    value_min: float,
    value_max: float,
    title: str,
    subtitle: str,
    x_column: str,
    y_column: str,
) -> list[str]:
    pad_left = 54
    pad_top = 72
    plot_width = width - 82
    plot_height = height - 114
    plot_x = x0 + pad_left
    plot_y = y0 + pad_top
    span = max(value_max - value_min, 1e-6)
    parts = [
        f'<rect x="{x0}" y="{y0}" width="{width}" height="{height}" rx="18" fill="{CARD_FILL}" stroke="{CARD_STROKE}" />',
        f'<text x="{x0 + 20}" y="{y0 + 28}" font-size="18" font-family="Georgia, serif" fill="{TEXT}">{escape(title)}</text>',
        f'<text x="{x0 + 20}" y="{y0 + 48}" font-size="11" font-family="Helvetica, Arial, sans-serif" fill="{MUTED}">{escape(subtitle)}</text>',
    ]
    parts.append(f'<rect x="{plot_x}" y="{plot_y}" width="{plot_width}" height="{plot_height}" fill="#fafaf9" stroke="{GRID}" />')
    if x_column == "actual" and y_column == "predicted_mean":
        parts.append(
            f'<line x1="{plot_x}" y1="{plot_y + plot_height}" x2="{plot_x + plot_width}" y2="{plot_y}" stroke="{GRID}" stroke-width="1.5" stroke-dasharray="5 5" />'
        )
    for _, row in frame.iterrows():
        x_value = float(row[x_column])
        y_value = float(row[y_column])
        cx = plot_x + ((x_value - value_min) / span) * plot_width
        cy = plot_y + plot_height - ((y_value - value_min) / span) * plot_height
        parts.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="3.2" fill="#b45309" opacity="0.8" />')
    parts.append(
        f'<text x="{plot_x}" y="{plot_y + plot_height + 22}" font-size="10" font-family="Helvetica, Arial, sans-serif" fill="{MUTED}">{escape(x_column)}</text>'
    )
    parts.append(
        f'<text x="{plot_x - 26}" y="{plot_y + 12}" font-size="10" font-family="Helvetica, Arial, sans-serif" fill="{MUTED}" transform="rotate(-90 {plot_x - 26} {plot_y + 12})">{escape(y_column)}</text>'
    )
    return parts


def _calibration_card_svg(frame: pd.DataFrame, *, x0: int, y0: int, width: int, height: int, title: str) -> list[str]:
    pad_left = 54
    pad_top = 72
    plot_width = width - 82
    plot_height = height - 114
    plot_x = x0 + pad_left
    plot_y = y0 + pad_top
    parts = [
        f'<rect x="{x0}" y="{y0}" width="{width}" height="{height}" rx="18" fill="{CARD_FILL}" stroke="{CARD_STROKE}" />',
        f'<text x="{x0 + 20}" y="{y0 + 28}" font-size="18" font-family="Georgia, serif" fill="{TEXT}">{escape(title)}</text>',
        f'<text x="{x0 + 20}" y="{y0 + 48}" font-size="11" font-family="Helvetica, Arial, sans-serif" fill="{MUTED}">Nominal vs empirical interval coverage on the test split.</text>',
        f'<rect x="{plot_x}" y="{plot_y}" width="{plot_width}" height="{plot_height}" fill="#fafaf9" stroke="{GRID}" />',
        f'<line x1="{plot_x}" y1="{plot_y + plot_height}" x2="{plot_x + plot_width}" y2="{plot_y}" stroke="{GRID}" stroke-width="1.5" stroke-dasharray="5 5" />',
    ]
    point_coords: list[tuple[float, float]] = []
    for _, row in frame.iterrows():
        x_value = float(row["nominal_coverage"])
        y_value = float(row["empirical_coverage"])
        cx = plot_x + x_value * plot_width
        cy = plot_y + plot_height - y_value * plot_height
        point_coords.append((cx, cy))
    if point_coords:
        path = " ".join(
            f"{'M' if index == 0 else 'L'} {cx:.1f} {cy:.1f}"
            for index, (cx, cy) in enumerate(point_coords)
        )
        parts.append(f'<path d="{path}" fill="none" stroke="#b45309" stroke-width="2" />')
    for (cx, cy), (_, row) in zip(point_coords, frame.iterrows(), strict=True):
        parts.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="3.5" fill="#b45309" />')
        parts.append(
            f'<text x="{cx + 6:.1f}" y="{cy - 6:.1f}" font-size="10" font-family="Helvetica, Arial, sans-serif" fill="{TEXT}">{float(row["nominal_coverage"]):.2f}</text>'
        )
    return parts


def _metric_comparison_card_svg(
    frame: pd.DataFrame,
    *,
    dataset: str,
    x0: int,
    y0: int,
    width: int,
    height: int,
    y_min: float,
    y_max: float,
) -> list[str]:
    ordered = frame.sort_values(["series_order", "series_key"]).reset_index(drop=True)
    plot_x = x0 + 48
    context = str(ordered.iloc[0].get("comparison_context", ""))
    paper_row = ordered.loc[ordered["series_key"] == "paper"]
    paper_title = str(paper_row.iloc[0]["paper_source_title"]) if not paper_row.empty else "Paper context unavailable for this metric."
    context_lines = _wrap_text(context, 58)[:2]
    paper_title_lines = _wrap_text(paper_title, 58)[:2]
    plot_y = y0 + 88 + 12 * max(0, len(context_lines) - 1) + 12 * max(0, len(paper_title_lines) - 1)
    plot_width = width - 76
    plot_height = 120
    zero_y = plot_y + plot_height - ((0.0 - y_min) / max(y_max - y_min, 1e-6)) * plot_height
    bars = [
        (
            str(row["series_label"]),
            float(row["value"]),
            str(row.get("series_color", "")) or SERIES_COLORS.get(str(row["series_key"]), "#475569"),
        )
        for _, row in ordered.iterrows()
        if pd.notna(row.get("value", pd.NA))
    ]
    bar_gap = 12 if len(bars) >= 5 else 16
    available_width = plot_width - 20 - bar_gap * max(0, len(bars) - 1)
    bar_width = max(36, min(54, int(available_width / max(1, len(bars)))))
    left_pad = max(8, int((plot_width - (len(bars) * bar_width + max(0, len(bars) - 1) * bar_gap)) / 2))
    parts = [
        f'<rect x="{x0}" y="{y0}" width="{width}" height="{height}" rx="18" fill="{CARD_FILL}" stroke="{CARD_STROKE}" />',
        f'<text x="{x0 + 20}" y="{y0 + 28}" font-size="18" font-family="Georgia, serif" fill="{TEXT}">{escape(dataset)}</text>',
    ]
    text_y = y0 + 48
    for line_index, line in enumerate(context_lines or [""]):
        parts.append(
            f'<text x="{x0 + 20}" y="{text_y + line_index * 12}" font-size="11" font-family="Helvetica, Arial, sans-serif" fill="{MUTED}">{escape(line)}</text>'
        )
    paper_base_y = text_y + max(1, len(context_lines)) * 12 + 4
    for line_index, line in enumerate(paper_title_lines or [""]):
        parts.append(
            f'<text x="{x0 + 20}" y="{paper_base_y + line_index * 12}" font-size="10" font-family="Helvetica, Arial, sans-serif" fill="{MUTED}">{escape(line)}</text>'
        )
    for step in range(5):
        fraction = step / 4.0
        y_value = y_min + (y_max - y_min) * fraction
        y = plot_y + plot_height - plot_height * fraction
        parts.append(f'<line x1="{plot_x}" y1="{y:.1f}" x2="{plot_x + plot_width}" y2="{y:.1f}" stroke="{GRID}" stroke-width="1" />')
        parts.append(
            f'<text x="{plot_x - 8}" y="{y + 4:.1f}" text-anchor="end" font-size="10" font-family="Helvetica, Arial, sans-serif" fill="{MUTED}">{y_value:.2f}</text>'
        )
    parts.append(f'<line x1="{plot_x}" y1="{zero_y:.1f}" x2="{plot_x + plot_width}" y2="{zero_y:.1f}" stroke="{TEXT}" stroke-width="1.4" />')
    for index, (label, value, color) in enumerate(bars):
        span = max(y_max - y_min, 1e-6)
        bar_height = abs(value - 0.0) / span * plot_height
        x = plot_x + left_pad + index * (bar_width + bar_gap)
        if value >= 0.0:
            y = zero_y - bar_height
        else:
            y = zero_y
        parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_width}" height="{bar_height:.1f}" rx="8" fill="{color}" opacity="0.92" />')
        parts.append(
            f'<text x="{x + bar_width / 2:.1f}" y="{plot_y + plot_height + 20}" text-anchor="middle" font-size="11" font-family="Helvetica, Arial, sans-serif" fill="{TEXT}">{escape(label)}</text>'
        )
        value_y = y - 8 if value >= 0.0 else y + bar_height + 14
        parts.append(
            f'<text x="{x + bar_width / 2:.1f}" y="{value_y:.1f}" text-anchor="middle" font-size="10" font-family="Helvetica, Arial, sans-serif" fill="{TEXT}">{value:.3f}</text>'
        )
    note_lines = _wrap_text(str(paper_row.iloc[0]["paper_note"]) if not paper_row.empty else "No numeric literature value was attached for this metric.", 60)
    for line_index, line in enumerate(note_lines[:3]):
        parts.append(
            f'<text x="{x0 + 20}" y="{y0 + height - 44 + line_index * 13}" font-size="10" font-family="Helvetica, Arial, sans-serif" fill="{MUTED}">{escape(line)}</text>'
        )
    return parts


def _metric_comparison_title(metric_label: str, comparison_key: str) -> str:
    if comparison_key == "frontier":
        return f"{metric_label} frontier"
    return f"{metric_label} comparison"


def _metric_comparison_filename(metric_key: str, comparison_key: str) -> str:
    if comparison_key == "frontier":
        return f"{metric_key}_frontier_comparison.svg"
    return f"{metric_key}_comparison.svg"


def _default_metric_comparison_subtitle(comparison_key: str) -> str:
    if comparison_key == "frontier":
        return "This mixed-family frontier adds the coordinate-backed `sdf_geom` and `moladt_geom` rows and allows geometry models when that improves the representation."
    return "Gray is paper context, orange is the raw SMILES string baseline, and teal is the typed MolADT baseline built from the same boundary strings."


def _series_order_for_key(series_key: str) -> int:
    order = {
        "paper": 0,
        "smiles": 1,
        "moladt": 2,
        "sdf_geom": 3,
        "moladt_geom": 4,
    }
    return order.get(series_key, len(order))
