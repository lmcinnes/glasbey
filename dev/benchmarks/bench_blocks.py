"""Per-stage timings for block palettes against the Phase 6 targets.

Run from the repository root:  uv run python dev/benchmarks/bench_blocks.py [--repeats N]
Targets (median, one core of a GitHub Actions runner, default parameters):
  one theme arc (n <= 10)              < 0.15 s
  create_block_palette([10,10,3x6])    < 2 s
  create_block_palette([5] * 20)       < 6 s
  select + allocate + refine, 20 blocks < 0.5 s
"""

import argparse
import statistics
import time
import warnings

import numpy as np

import glasbey
from glasbey._blocks import make_context, plan_blocks, refine, reserved_sweeps, select_bases
from glasbey._curves import _cam02ucs_to_jch
from glasbey._glasbey import _build_block_palette_with_diagnostics
from glasbey._theme import fit_theme_arc


def median_time(function, repeats):
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        function()
        times.append(time.perf_counter() - start)
    return statistics.median(times)


def bench_theme_arc(repeats):
    context = make_context()
    jch = _cam02ucs_to_jch(context.grid.colors)
    pool = context.grid.colors[(jch[:, 0] > 45) & (jch[:, 0] < 65) & (jch[:, 1] > 20)]
    bases = pool[np.random.default_rng(0).integers(len(pool), size=8)]

    def run():
        for index, base in enumerate(bases):
            fit_theme_arc(base, 3 + index, domain=context.domain, gamut=context.gamut,
                          grid=context.grid, hue_allowance=(15, 15), warn=False)

    return median_time(run, repeats) / len(bases)


def bench_selection(repeats):
    sizes = [10, 8, 6, 6, 4, 4, 3, 3, 3, 3, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1]

    def run():
        context = make_context()
        specs = plan_blocks(sizes, context)
        reserved = reserved_sweeps(specs, context)
        footprints, info = select_bases(specs, context, reserved, 1)
        refine(specs, footprints, info, context, reserved, 1)

    return median_time(run, repeats)


def bench_palette(sizes, repeats):
    stages = []

    def run():
        _, diagnostics = _build_block_palette_with_diagnostics(sizes)
        stages.append(diagnostics["timings"])

    total = median_time(run, repeats)
    per_stage = {key: statistics.median(stage[key] for stage in stages) for key in stages[0]}
    return total, per_stage


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    warnings.simplefilter("ignore")
    glasbey.create_block_palette([3, 2])  # warm numba and caches
    rows = [("theme arc (per arc, n <= 10)", bench_theme_arc(args.repeats), 0.15, None),
            ("select + allocate + refine, 20 blocks", bench_selection(args.repeats), 0.5, None)]
    for sizes, target in (([10, 10, 3, 3, 3, 3, 3, 3], 2.0), ([5] * 20, 6.0)):
        total, stages = bench_palette(sizes, args.repeats)
        label = f"create_block_palette({sizes if len(sizes) < 10 else '[5] * 20'})"
        rows.append((label, total, target, stages))
    for label, value, target, stages in rows:
        status = "ok" if value < target else "SLOW"
        print(f"{label:<55} {value:7.3f} s  (target {target:g} s)  {status}")
        if stages:
            print("    " + "  ".join(f"{key}={seconds:.3f}" for key, seconds in stages.items()))


if __name__ == "__main__":
    main()
