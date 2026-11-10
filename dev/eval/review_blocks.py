"""Side-by-side aesthetic review of block palettes: legacy (0.4.0) versus current.

Run from the repository root:

    uv run python dev/eval/review_blocks.py --out dev/eval/results/review.png

Each configuration shows the legacy palette above the current one at the
given bounds, with block boundaries marked, so the two can be compared by eye.
"""

import argparse
import warnings
from pathlib import Path

import numpy as np
from matplotlib.colors import to_rgb

import glasbey

CASES = [
    ("[4, 3, 3, 2]", [4, 3, 3, 2], {}),
    ("[2] * 6", [2] * 6, {}),
    ("[3] * 8", [3] * 8, {}),
    ("[5, 4, 3, 3, 2, 2, 1]", [5, 4, 3, 3, 2, 2, 1], {}),
    ("[6] * 4", [6] * 4, {}),
    ("[10, 10, 3 x 6]", [10, 10, 3, 3, 3, 3, 3, 3], {}),
    ("[12, 8, 3 x 4, 2, 2]", [12, 8, 3, 3, 3, 3, 2, 2], {}),
    ("[4, 4, 4] L 40-70", [4, 4, 4], {"lightness_bounds": (40, 70)}),
    ("[5, 4, 3, 3, 2, 2, 1] C 0-40", [5, 4, 3, 3, 2, 2, 1], {"chroma_bounds": (0, 40)}),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default="dev/eval/results/review.png")
    args = parser.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    warnings.simplefilter("ignore")
    fig, axes = plt.subplots(len(CASES), 1, figsize=(12, 1.15 * len(CASES) + 0.4))
    for ax, (label, sizes, options) in zip(axes, CASES):
        legacy = glasbey.create_block_palette(sizes, theme_method="legacy", **options)
        current = glasbey.create_block_palette(sizes, **options)
        width = max(len(legacy), 1)
        image = np.ones((2 * 10 + 3, width * 10, 3))
        for row, palette in ((0, legacy), (13, current)):
            for index, color in enumerate(palette):
                image[row:row + 10, index * 10:index * 10 + 10] = to_rgb(color)
        for edge in np.cumsum(sizes)[:-1]:
            image[:, edge * 10 - 1:edge * 10 + 1] = 1.0
        ax.imshow(image, aspect="auto", interpolation="nearest")
        ax.set_xticks([])
        ax.set_yticks([5, 18])
        ax.set_yticklabels(["legacy", "current"], fontsize=8)
        ax.set_title(label, fontsize=9, loc="left")
    fig.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=100)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
