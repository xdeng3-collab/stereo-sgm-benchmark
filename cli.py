#!/usr/bin/env python3
"""Command line entry point.

    python cli.py evaluate                 compare matchers on a synthetic pair
    python cli.py sweep --out results/     parameter grid, CSV plus frontier plot
    python cli.py middlebury <scene-dir>   evaluate on a real Middlebury scene
"""

from __future__ import annotations

import argparse
from pathlib import Path

from stereo_bench.datasets import load_middlebury, random_dot_pair
from stereo_bench.matchers import OpenCVBM, OpenCVSGBM, SGMParams, time_match
from stereo_bench.metrics import evaluate
from stereo_bench.sweep import grid, pareto_frontier, plot_frontier, run_sweep, write_csv

HEADER = f"{'matcher':<14}{'MAE':>8}{'RMSE':>8}{'bad1.0':>9}{'bad3.0':>9}{'density':>9}{'ms':>9}"


def _line(name: str, metrics, elapsed_s: float) -> str:
    return (
        f"{name:<14}{metrics.mae:8.3f}{metrics.rmse:8.3f}"
        f"{metrics.bad_1_0:9.3f}{metrics.bad_3_0:9.3f}"
        f"{metrics.density:9.3f}{elapsed_s * 1000:9.1f}"
    )


def cmd_evaluate(args: argparse.Namespace) -> int:
    pair = random_dot_pair(
        args.width, args.height, seed=args.seed,
        max_disparity=args.max_disparity, blank_fraction=args.blank_fraction,
    )
    params = SGMParams(num_disparities=args.max_disparity, block_size=args.block_size)
    print(f"pair: {pair.name}\n")
    print(HEADER)
    for factory in (OpenCVSGBM, OpenCVBM):
        result = time_match(factory(params), pair.left, pair.right, repeats=args.repeats)
        metrics = evaluate(result.disparity, pair.truth, truth_mask=pair.occlusion)
        print(_line(result.matcher, metrics, result.elapsed_s))
    return 0


def cmd_sweep(args: argparse.Namespace) -> int:
    pairs = [
        random_dot_pair(args.width, args.height, seed=seed, max_disparity=args.max_disparity,
                        blank_fraction=args.blank_fraction)
        for seed in range(args.pairs)
    ]
    settings = grid(
        num_disparities=[args.max_disparity],
        block_size=[3, 5, 7, 9],
        p1=[8 * 25, 8 * 49],
        p2=[32 * 25, 32 * 49, 128 * 25],
        uniqueness_ratio=[5, 10, 15],
        mode_hh=[False, True],
    )
    print(f"{len(settings)} settings x {len(pairs)} pairs")
    rows = run_sweep(pairs, OpenCVSGBM, settings, repeats=args.repeats)

    out = Path(args.out)
    write_csv(rows, out / "sweep.csv")
    plot_frontier(rows, out / "frontier.png", title="OpenCV SGBM: accuracy vs latency")

    print(f"\nwrote {out / 'sweep.csv'} and {out / 'frontier.png'}\n")
    print("Pareto frontier:")
    print(f"{'ms':>8}{'MAE':>8}{'bad1.0':>9}{'density':>9}  params")
    for row in pareto_frontier(rows):
        params = ", ".join(f"{k}={v}" for k, v in row.params.items() if k != "num_disparities")
        print(f"{row.elapsed_s * 1000:8.1f}{row.metrics.mae:8.3f}"
              f"{row.metrics.bad_1_0:9.3f}{row.metrics.density:9.3f}  {params}")
    return 0


def cmd_middlebury(args: argparse.Namespace) -> int:
    pair = load_middlebury(Path(args.scene))
    if pair.truth is None:
        print(f"{pair.name} has no disp0.pfm; nothing to score against")
        return 1
    params = SGMParams(num_disparities=args.max_disparity, block_size=args.block_size)
    print(f"pair: {pair.name} {pair.shape}\n")
    print(HEADER)
    result = time_match(OpenCVSGBM(params), pair.left, pair.right, repeats=args.repeats)
    metrics = evaluate(result.disparity, pair.truth)
    print(_line(result.matcher, metrics, result.elapsed_s))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--max-disparity", type=int, default=64)
    parser.add_argument("--block-size", type=int, default=5)
    parser.add_argument("--blank-fraction", type=float, default=0.15,
                        help="size of the textureless patch, as a fraction of the image")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=3)

    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("evaluate", help="compare matchers on a synthetic pair").set_defaults(func=cmd_evaluate)

    sweep = sub.add_parser("sweep", help="parameter grid with CSV and frontier plot")
    sweep.add_argument("--out", default="results")
    sweep.add_argument("--pairs", type=int, default=2)
    sweep.set_defaults(func=cmd_sweep)

    mb = sub.add_parser("middlebury", help="evaluate a Middlebury 2014 scene directory")
    mb.add_argument("scene")
    mb.set_defaults(func=cmd_middlebury)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
