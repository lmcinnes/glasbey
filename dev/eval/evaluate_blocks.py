"""Evaluate block palettes: legacy (0.4.0) versus the constrained-arc method.

Run from the repository root:

    uv run python dev/eval/evaluate_blocks.py --out dev/eval/results [--quick]

Writes ``metrics.csv`` (one row per palette) and PNG galleries of swatch rows
and hue-lightness plots (one dot per colour, x for an arc block's base) per
bounds setting. ``--quick`` runs a small subset for smoke testing.
"""

import argparse
import csv
import itertools
import time
import warnings
from pathlib import Path

import numpy as np
from colorspacious import cspace_convert
from matplotlib.colors import to_rgb

import glasbey
from glasbey._glasbey import _build_block_palette_with_diagnostics
from glasbey._theme import discounted_distance, hue_offset

CONFIGS = {
    "3x8": [3] * 8,
    "6x6": [6] * 6,
    "10x4": [10] * 4,
    "10-10-3x6": [10, 10, 3, 3, 3, 3, 3, 3],
    "12-8-3x4-2x2": [12, 8, 3, 3, 3, 3, 2, 2],
    "10-6-4-3-3-2": [10, 6, 4, 3, 3, 2],
    "5x20": [5] * 20,
    "1-5-3-4-2": [1, 5, 3, 4, 2],
}
BOUNDS = {
    "L10-90_C0-80": dict(lightness_bounds=(10, 90), chroma_bounds=(0, 80)),
    "L30-80_C0-80": dict(lightness_bounds=(30, 80), chroma_bounds=(0, 80)),
    "L40-70_C0-80": dict(lightness_bounds=(40, 70), chroma_bounds=(0, 80)),
    "L10-90_C30-80": dict(lightness_bounds=(10, 90), chroma_bounds=(30, 80)),
    "L30-80_C30-80": dict(lightness_bounds=(30, 80), chroma_bounds=(30, 80)),
    "L40-70_C30-80": dict(lightness_bounds=(40, 70), chroma_bounds=(30, 80)),
    "L10-90_H300-60": dict(lightness_bounds=(10, 90), chroma_bounds=(0, 80), hue_bounds=(300, 60)),
}
VARIANTS = {
    "plain": dict(colorblind_safe=False, optimize_palette=False),
    "optimized": dict(colorblind_safe=False, optimize_palette=True),
    "cvd": dict(colorblind_safe=True, optimize_palette=False),
    "cvd_optimized": dict(colorblind_safe=True, optimize_palette=True),
}
CVD = {"name": "sRGB1+CVD", "cvd_type": "deuteranomaly", "severity": 50.0}
LONG_BLOCK_MIN = 6


def split(colors, sizes):
    edges = np.concatenate([[0], np.cumsum(sizes)])
    return [colors[a:b] for a, b in zip(edges[:-1], edges[1:])]


def between(blocks, w_j):
    best = np.inf
    for i in range(len(blocks)):
        for j in range(i + 1, len(blocks)):
            best = min(best, float(discounted_distance(blocks[i], blocks[j], w_j).min()))
    return best


def metrics(palette, sizes, sweeps=None):
    rgb = np.array([to_rgb(color) for color in palette])
    cam = cspace_convert(rgb, "sRGB1", "CAM02-UCS")
    jch = cspace_convert(rgb, "sRGB1", "JCh")
    cvd = cspace_convert(np.clip(cspace_convert(rgb, CVD, "sRGB1"), 0, 1), "sRGB1", "CAM02-UCS")
    blocks, jch_blocks = split(cam, sizes), split(jch, sizes)
    steps = [np.linalg.norm(np.diff(block, axis=0), axis=1) for block in blocks if len(block) > 1]
    all_steps = np.concatenate(steps) if steps else np.zeros(0)
    cvs = [s.std() / s.mean() for s in steps if len(s) > 1 and s.mean() > 0]
    spreads, over = [], []
    for index, block in enumerate(jch_blocks):
        chromatic = block[block[:, 1] > 15]
        if len(chromatic) >= 2:
            offsets = hue_offset(chromatic[:, 2], chromatic[0, 2])
            spread = float(offsets.max() - offsets.min())
            spreads.append(spread)
            if sweeps is not None:
                over.append(spread - sweeps[index])
    long_ab = [np.linalg.norm(np.diff(block[:, 1:], axis=0), axis=1)
               for block in blocks if len(block) >= LONG_BLOCK_MIN]
    within_max = float(all_steps.max()) if len(all_steps) else np.nan
    discounted = between(blocks, 0.35) if len(blocks) > 1 else np.nan
    return {
        "min_step": float(all_steps.min()) if len(all_steps) else np.nan,
        "median_step": float(np.median(all_steps)) if len(all_steps) else np.nan,
        "max_step_cv": float(max(cvs)) if cvs else 0.0,
        "monotone": bool(all(np.all(np.diff(b[:, 0]) > 0) for b in blocks)),
        "duplicates": len(palette) - len(set(palette)),
        "max_hue_spread": float(max(spreads)) if spreads else 0.0,
        "max_spread_minus_sweep": float(max(over)) if over else np.nan,
        "min_between_full": between(blocks, 1.0) if len(blocks) > 1 else np.nan,
        "min_between_discounted": discounted,
        "separability": discounted / within_max if within_max and within_max > 0 else np.nan,
        "long_ab_step": float(np.concatenate(long_ab).mean()) if long_ab else np.nan,
        "cvd_min_between_full": between(split(cvd, sizes), 1.0) if len(blocks) > 1 else np.nan,
        "cvd_min_between_discounted": between(split(cvd, sizes), 0.35) if len(blocks) > 1 else np.nan,
    }


def run_one(method, sizes, options):
    start = time.perf_counter()
    if method == "legacy":
        palette = glasbey.create_block_palette(sizes, theme_method="legacy", **options)
        return palette, {"total": time.perf_counter() - start}, None, None
    palette, diagnostics = _build_block_palette_with_diagnostics(sizes, **options)
    timings = dict(diagnostics["timings"], total=time.perf_counter() - start)
    bases = [theme.base for theme in diagnostics["themes"]]
    return palette, timings, diagnostics["sweeps"], bases


def gallery(path, title, rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(len(rows), 2, figsize=(14, 1.6 * len(rows) + 0.6),
                             gridspec_kw={"width_ratios": [3, 1]}, squeeze=False)
    for (label, palette, sizes, bases), (swatch, scatter) in zip(rows, axes):
        swatch.imshow([[to_rgb(color) for color in palette]], aspect="auto")
        for edge in np.cumsum(sizes)[:-1]:
            swatch.axvline(edge - 0.5, color="white", linewidth=2)
        swatch.set_yticks([])
        swatch.set_xticks([])
        swatch.set_ylabel(label, rotation=0, ha="right", va="center", fontsize=8)
        jch = cspace_convert([to_rgb(color) for color in palette], "sRGB1", "JCh")
        scatter.scatter(jch[:, 2], jch[:, 0], c=[to_rgb(color) for color in palette], s=14)
        if bases is not None:
            base_jch = cspace_convert(np.asarray(bases), "CAM02-UCS", "JCh")
            scatter.scatter(base_jch[:, 2], base_jch[:, 0], marker="x", c="black", s=20)
        scatter.set_xlim(0, 360)
        scatter.set_ylim(0, 100)
        scatter.tick_params(labelsize=6)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=90)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default="dev/eval/results")
    parser.add_argument("--quick", action="store_true", help="run a small subset")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    configs, bounds, variants = CONFIGS, BOUNDS, VARIANTS
    if args.quick:
        configs = {key: CONFIGS[key] for key in ("10-10-3x6", "1-5-3-4-2")}
        bounds = {key: BOUNDS[key] for key in ("L10-90_C0-80", "L40-70_C0-80")}
        variants = {key: VARIANTS[key] for key in ("plain", "cvd")}
    warnings.simplefilter("ignore")
    rows, galleries = [], {}
    for (bound_name, bound), (config_name, sizes), (variant_name, variant) in itertools.product(
        bounds.items(), configs.items(), variants.items()
    ):
        options = dict(bound, **variant)
        for method in ("legacy", "arc"):
            palette, timings, sweeps, bases = run_one(method, sizes, options)
            row = {"bounds": bound_name, "config": config_name, "variant": variant_name,
                   "method": method}
            row.update(metrics(palette, sizes, sweeps))
            row.update({f"time_{key}": value for key, value in timings.items()})
            rows.append(row)
            if variant_name == "plain":
                galleries.setdefault(bound_name, []).append(
                    (f"{config_name}\n{method}", palette, sizes, bases))
            print(f"{bound_name:16} {config_name:14} {variant_name:13} {method:6} "
                  f"sep={row['min_between_discounted']:6.2f} cv={row['max_step_cv']:.3f} "
                  f"t={timings['total']:.2f}s", flush=True)
    fields = sorted({key for row in rows for key in row},
                    key=lambda key: (not key in ("bounds", "config", "variant", "method"), key))
    with open(out / "metrics.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    for bound_name, gallery_rows in galleries.items():
        gallery(out / f"gallery_{bound_name}.png", bound_name, gallery_rows)
    print(f"wrote {len(rows)} rows and {len(galleries)} galleries to {out}")


if __name__ == "__main__":
    main()
