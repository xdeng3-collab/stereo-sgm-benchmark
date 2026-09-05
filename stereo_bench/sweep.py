"""Parameter sweeps and the accuracy-latency frontier.

A single (accuracy, latency) pair says almost nothing, because any matcher can
be made more accurate by being slower. What a robot actually needs is the
frontier: at the latency budget the platform allows, what is the best accuracy
available, and which parameters get there.
"""

from __future__ import annotations

import csv
import itertools
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

from .datasets import StereoPair
from .matchers import MatchResult, SGMParams, time_match
from .metrics import DisparityMetrics, evaluate


@dataclass(frozen=True)
class SweepRow:
    pair: str
    matcher: str
    params: dict[str, Any]
    metrics: DisparityMetrics
    elapsed_s: float

    def flat(self) -> dict[str, Any]:
        row: dict[str, Any] = {"pair": self.pair, "matcher": self.matcher}
        row.update({f"param_{k}": v for k, v in self.params.items()})
        row.update(self.metrics.as_dict())
        row["elapsed_ms"] = self.elapsed_s * 1000.0
        return row


def grid(**axes: Sequence[Any]) -> list[dict[str, Any]]:
    """Cartesian product of named parameter axes, as a list of kwargs dicts."""
    names = list(axes)
    return [dict(zip(names, values)) for values in itertools.product(*(axes[n] for n in names))]


def run_sweep(
    pairs: Iterable[StereoPair],
    build_matcher: Callable[[SGMParams], Any],
    settings: Sequence[dict[str, Any]],
    *,
    repeats: int = 3,
) -> list[SweepRow]:
    """Evaluate one matcher across a parameter grid on every pair."""
    rows: list[SweepRow] = []
    for pair in pairs:
        if pair.truth is None:
            raise ValueError(f"pair {pair.name} has no ground truth to score against")
        for setting in settings:
            params = SGMParams(**setting)
            result: MatchResult = time_match(
                build_matcher(params), pair.left, pair.right, repeats=repeats
            )
            metrics = evaluate(result.disparity, pair.truth, truth_mask=pair.occlusion)
            rows.append(
                SweepRow(
                    pair=pair.name,
                    matcher=result.matcher,
                    params=setting,
                    metrics=metrics,
                    elapsed_s=result.elapsed_s,
                )
            )
    return rows


def pareto_frontier(rows: Sequence[SweepRow]) -> list[SweepRow]:
    """Rows that no other row beats on both accuracy and latency.

    Rows whose density is zero, or whose error is undefined, are dropped: a
    matcher that estimates nothing is not on anyone's frontier.
    """
    usable = [r for r in rows if r.metrics.evaluated > 0 and r.metrics.mae == r.metrics.mae]
    frontier: list[SweepRow] = []
    for row in usable:
        dominated = any(
            other.metrics.mae <= row.metrics.mae
            and other.elapsed_s <= row.elapsed_s
            and (other.metrics.mae < row.metrics.mae or other.elapsed_s < row.elapsed_s)
            for other in usable
        )
        if not dominated:
            frontier.append(row)
    return sorted(frontier, key=lambda r: r.elapsed_s)


def write_csv(rows: Sequence[SweepRow], path: Path) -> None:
    if not rows:
        raise ValueError("nothing to write")
    flat = [row.flat() for row in rows]
    fieldnames = list(flat[0])
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(flat)


def plot_frontier(rows: Sequence[SweepRow], path: Path, *, title: str = "") -> None:
    """Scatter every setting, and draw the frontier through the winners."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    usable = [r for r in rows if r.metrics.evaluated > 0 and r.metrics.mae == r.metrics.mae]
    if not usable:
        raise ValueError("no scorable rows to plot")

    frontier = pareto_frontier(usable)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.scatter(
        [r.elapsed_s * 1000 for r in usable],
        [r.metrics.mae for r in usable],
        s=18, alpha=0.45, label="all settings",
    )
    ax.plot(
        [r.elapsed_s * 1000 for r in frontier],
        [r.metrics.mae for r in frontier],
        marker="o", linewidth=1.6, label="Pareto frontier",
    )
    ax.set_xlabel("time per frame (ms)")
    ax.set_ylabel("disparity MAE (px)")
    ax.set_title(title or "Accuracy vs latency")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)
