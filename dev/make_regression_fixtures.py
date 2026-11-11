"""Regenerate regression fixtures from the *current* code.

Run from the repository root with ``uv run python dev/make_regression_fixtures.py``.
The fixtures pin ``create_colormap`` output and the 0.4.0 block palette output
(``theme_method="legacy"``). Only regenerate them deliberately.
"""

import json
from pathlib import Path

import numpy as np

import glasbey
from glasbey import _legacy_blocks

FIXTURES = Path(__file__).resolve().parents[1] / "glasbey" / "tests" / "fixtures"

COLORMAP_CASES = {
    "sequential": dict(seed_colors=["#bdd8e8", "#324555", "#64809b"], grid_size=12, max_iter=80),
    "diverging": dict(seed_colors=["#4a6fa5", "#efefef", "#b5603a"], mode="diverging",
                      grid_size=12, max_iter=80),
    "grey_vibrant": dict(seed_colors=["#333333", "#dddddd"], grid_size=12, arc_length_factor=1.3,
                         smoothness=3, chroma_weight=5, max_iter=80),
}

BLOCK_CASES = {
    "default_mixed": dict(block_sizes=[5, 5, 3, 2, 2, 1]),
    "jch_grid": dict(block_sizes=[1, 5, 3, 4, 2], grid_space="JCh"),
    "tight_no_opt": dict(block_sizes=[4, 4, 3], lightness_bounds=(40, 70), optimize_palette=False),
    "colorblind": dict(block_sizes=[4, 4, 4, 4], colorblind_safe=True, cvd_severity=100.0),
}

EXTEND_CASES = {
    "extend_tab10": dict(palette="tab10", new_block_sizes=[3, 2]),
}

THEME_CASES = {
    "theme_default": dict(base_color="#4a7ab0", palette_size=5),
    "theme_seven": dict(base_color="#b05a4a", palette_size=7),
}


def colormap_fixtures(create_colormap):
    return {name: create_colormap(**case).sample(17)[1].tolist() for name, case in COLORMAP_CASES.items()}


def block_fixtures(create_block, extend_block, create_theme, **extra):
    result = {name: create_block(**case, **extra) for name, case in BLOCK_CASES.items()}
    result.update({name: extend_block(**case, **extra) for name, case in EXTEND_CASES.items()})
    result.update({name: list(create_theme(**case)) for name, case in THEME_CASES.items()})
    return result


if __name__ == "__main__":
    FIXTURES.mkdir(exist_ok=True)
    (FIXTURES / "colormap_regression.json").write_text(
        json.dumps(colormap_fixtures(glasbey.create_colormap), indent=1)
    )
    (FIXTURES / "legacy_blocks_0_4_0.json").write_text(json.dumps(block_fixtures(
        _legacy_blocks.create_block_palette, _legacy_blocks.extend_block_palette,
        _legacy_blocks.create_theme_palette,
    ), indent=1))
    print("wrote fixtures to", FIXTURES)
