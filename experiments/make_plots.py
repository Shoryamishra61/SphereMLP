"""Dependency-free SVG diagnostics for frozen benchmark records."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def layerwise_svg(raw_path: Path, output: Path) -> None:
    """Write a log-scale layerwise-MSE curve without a plotting dependency."""
    with raw_path.open(newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle) if row["status"] == "ok"]
    names = sorted({row["estimator_name"] for row in rows})
    series = {
        name: np.mean(
            [json.loads(row["per_layer_mse"]) for row in rows if row["estimator_name"] == name],
            axis=0,
        )
        for name in names
    }
    width, height, left, top, right, bottom = 900, 520, 80, 35, 25, 70
    plot_w, plot_h = width - left - right, height - top - bottom
    all_values = np.concatenate([np.asarray(value) for value in series.values()])
    lo, hi = np.log10(max(float(np.min(all_values)), 1e-15)), np.log10(float(np.max(all_values)))
    colors = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00"]
    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<path d="M {left} {top} V {height - bottom} H {width - right}" stroke="#222" fill="none"/>',
        '<text x="20" y="25" font-family="sans-serif" font-size="17">T06 validation: layerwise MSE (log scale)</text>',
        f'<text x="{width // 2}" y="{height - 20}" font-family="sans-serif" font-size="13">layer</text>',
        f'<text x="18" y="{height // 2}" font-family="sans-serif" font-size="13" transform="rotate(-90 18 {height // 2})">mean squared error</text>',
    ]
    layers = len(next(iter(series.values())))
    for index, (name, values) in enumerate(series.items()):
        points = []
        for layer, value in enumerate(values):
            x = left + plot_w * layer / max(layers - 1, 1)
            y = top + plot_h * (1.0 - (np.log10(max(value, 1e-15)) - lo) / max(hi - lo, 1e-12))
            points.append(f"{x:.1f},{y:.1f}")
        color = colors[index % len(colors)]
        elements.append(
            f'<polyline points="{" ".join(points)}" fill="none" stroke="{color}" stroke-width="2"/>'
        )
        elements.append(
            f'<text x="{left + 10}" y="{top + 24 + 19 * index}" font-family="sans-serif" font-size="13" fill="{color}">{name}</text>'
        )
    for fraction in (0.0, 0.5, 1.0):
        exponent = hi - fraction * (hi - lo)
        y = top + fraction * plot_h
        elements.append(f'<path d="M {left - 4} {y:.1f} H {width - right}" stroke="#ddd"/>')
        elements.append(
            f'<text x="{left - 66}" y="{y + 4:.1f}" font-family="monospace" font-size="11">1e{exponent:.1f}</text>'
        )
    elements.append("</svg>")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(elements), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--layerwise-output", type=Path, required=True)
    args = parser.parse_args()
    layerwise_svg(args.raw, args.layerwise_output)


if __name__ == "__main__":
    main()
