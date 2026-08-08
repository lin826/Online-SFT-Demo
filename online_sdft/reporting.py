"""Aggregation, qualitative selection, and plotting.

This module reads completed chronological traces. It never participates in an
action or update, which keeps evaluation-only oracle fields out of methods.
"""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from .config import METHODS, PHASE_LENGTH


def mean_ci(values: list[float]) -> dict:
    array = np.asarray(values, dtype=float)
    if len(array) == 1:
        return {"mean": float(array[0]), "std": 0.0, "ci95": 0.0}
    std = float(array.std(ddof=1))
    return {
        "mean": float(array.mean()),
        "std": std,
        "ci95": float(1.96 * std / math.sqrt(len(array))),
    }


def summarize_metrics(metrics: list[dict]) -> dict:
    metric_names = [
        key
        for key in metrics[0]
        if key not in {"seed", "method"}
    ]
    return {
        method: {
            metric: mean_ci(
                [
                    float(row[metric])
                    for row in metrics
                    if row["method"] == method
                ]
            )
            for metric in metric_names
        }
        for method in METHODS
    }


def find_qualitative_examples(
    rollouts: list[dict],
    limit: int = 8,
) -> list[dict]:
    """Select later steps uniquely solved by Online-SDFT."""
    grouped = defaultdict(dict)
    for row in rollouts:
        grouped[(row["seed"], row["t"])][row["method"]] = row

    qualitative = []
    for (seed, step), rows in sorted(grouped.items()):
        if (
            step > PHASE_LENGTH
            and len(rows) == len(METHODS)
            and rows["Online-SDFT"]["correct_online"] == 1
            and sum(
                rows[method]["correct_online"]
                for method in METHODS[:-1]
            )
            == 0
        ):
            qualitative.append(
                {
                    "seed": seed,
                    "t": step,
                    "regime": rows["Online-SDFT"]["regime"],
                    "category": rows["Online-SDFT"]["category"],
                    "oracle_action_scoring_only": rows["Online-SDFT"][
                        "oracle_action_scoring_only"
                    ],
                    "methods": {
                        method: {
                            "action": rows[method]["action"],
                            "feedback": rows[method]["feedback"],
                            "teacher_rollout": rows[method][
                                "teacher_rollout"
                            ],
                        }
                        for method in METHODS
                    },
                }
            )
        if len(qualitative) >= limit:
            break
    return qualitative


def write_compact_results(
    output_dir: Path,
    config: dict,
    metrics: list[dict],
    rollouts: list[dict],
) -> dict:
    summary = summarize_metrics(metrics)
    qualitative = find_qualitative_examples(rollouts)
    (output_dir / "qualitative_examples.json").write_text(
        json.dumps(qualitative, indent=2) + "\n"
    )
    payload = {
        "config": config,
        "summary": summary,
        "qualitative_examples": len(qualitative),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(payload, indent=2) + "\n"
    )
    return summary


# Baselines are muted so the two teacher-driven learners and REINFORCE carry
# the colour; the same mapping is reused by every figure and by the website.
METHOD_COLORS = {
    "Base": "#c3c9d2",
    "ICL": "#a5adb9",
    "RAG": "#828d9c",
    "REINFORCE": "#cf7a3e",
    "Online-SFT": "#8a72c4",
    "Online-SDFT": "#1f5fd8",
}
# The three frozen baselines land on nearly the same numbers, so they also get
# distinct dash patterns to stay readable where the curves overlap.
METHOD_DASHES = {
    "Base": (0, ()),
    "ICL": (0, (5, 2)),
    "RAG": (0, (1.6, 1.8)),
    "REINFORCE": (0, ()),
    "Online-SFT": (0, ()),
    "Online-SDFT": (0, ()),
}
INK = "#1f2328"
MUTED = "#6b7280"
GRID = "#e6e8ec"
PHASE_LABELS = ("weekday", "on-call", "off-hours")


def _apply_plot_style(matplotlib) -> None:
    matplotlib.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [
                "Helvetica Neue",
                "Helvetica",
                "Arial",
                "DejaVu Sans",
            ],
            "font.size": 9,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#c8ccd3",
            "axes.linewidth": 0.8,
            "axes.labelcolor": MUTED,
            "axes.labelsize": 8.5,
            "axes.titlesize": 10,
            "axes.titlecolor": INK,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "text.color": INK,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "xtick.major.size": 3,
            "ytick.major.size": 3,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "legend.frameon": False,
            "legend.fontsize": 8.5,
            "savefig.facecolor": "white",
        }
    )


def _draw_phase_bands(axis, upper: float) -> None:
    """Shade the three preference regimes instead of drawing bare cut lines."""
    for index in range(3):
        start = index * PHASE_LENGTH
        end = start + PHASE_LENGTH
        if index % 2 == 1:
            axis.axvspan(start, end, color="#f4f5f7", zorder=0)
        axis.text(
            start + PHASE_LENGTH / 2,
            upper,
            PHASE_LABELS[index],
            ha="center",
            va="bottom",
            fontsize=7.5,
            color=MUTED,
        )


def write_figures(
    summary: dict,
    curves: list[dict],
    figure_dir: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    _apply_plot_style(matplotlib)
    import matplotlib.pyplot as plt

    figure_dir.mkdir(parents=True, exist_ok=True)
    ordered = list(METHODS)
    positions = np.arange(len(ordered))

    figure, axes = plt.subplots(1, 2, figsize=(8.2, 3.2), sharey=True)
    panels = (
        (
            axes[0],
            [100 * summary[m]["online_accuracy"]["mean"] for m in ordered],
            [100 * summary[m]["online_accuracy"]["ci95"] for m in ordered],
            "Online accuracy",
            "percent of decisions matching the best route  (higher is better)",
            "{:.1f}",
        ),
        (
            axes[1],
            [summary[m]["cum_regret"]["mean"] for m in ordered],
            [summary[m]["cum_regret"]["ci95"] for m in ordered],
            "Cumulative regret",
            "utility lost over 240 decisions  (lower is better)",
            "{:.1f}",
        ),
    )
    for axis, values, errors, title, subtitle, fmt in panels:
        axis.barh(
            positions,
            values,
            xerr=errors,
            height=0.6,
            color=[METHOD_COLORS[m] for m in ordered],
            error_kw={
                "ecolor": "#5b6572",
                "elinewidth": 0.9,
                "capsize": 2.5,
                "capthick": 0.9,
            },
            zorder=3,
        )
        headroom = max(v + e for v, e in zip(values, errors)) * 1.16
        for position, value, error in zip(positions, values, errors):
            axis.text(
                value + error + headroom * 0.02,
                position,
                fmt.format(value),
                va="center",
                ha="left",
                fontsize=8,
                color=INK,
            )
        axis.set_xlim(0, headroom)
        axis.set_title(title, loc="left", pad=14, fontweight="medium")
        axis.set_xlabel(subtitle, labelpad=7)
        axis.grid(axis="x", zorder=0)
        axis.set_axisbelow(True)
        axis.spines["left"].set_visible(False)
        axis.tick_params(axis="y", length=0)

    axes[0].set_yticks(positions, ordered)
    for label, method in zip(axes[0].get_yticklabels(), ordered):
        label.set_color(INK if method == "Online-SDFT" else MUTED)
    axes[0].invert_yaxis()
    figure.tight_layout(pad=0.6)
    figure.subplots_adjust(wspace=0.12)
    figure.savefig(
        figure_dir / "bandit_accuracy.png",
        dpi=220,
        bbox_inches="tight",
    )
    plt.close(figure)

    accuracy_by_step = defaultdict(list)
    regret_by_step = defaultdict(list)
    for row in curves:
        key = (row["method"], int(row["t"]))
        accuracy_by_step[key].append(float(row["cum_accuracy"]))
        regret_by_step[key].append(float(row["cum_regret"]))

    figure, axes = plt.subplots(1, 2, figsize=(8.2, 3.3))
    handles = []
    for method in ordered:
        steps = sorted(step for name, step in accuracy_by_step if name == method)
        samples = [np.asarray(accuracy_by_step[(method, s)]) for s in steps]
        mean = np.array([s.mean() for s in samples]) * 100
        ci = (
            np.array(
                [
                    0.0
                    if len(s) == 1
                    else 1.96 * s.std(ddof=1) / math.sqrt(len(s))
                    for s in samples
                ]
            )
            * 100
        )
        emphasis = method == "Online-SDFT"
        (line,) = axes[0].plot(
            steps,
            mean,
            color=METHOD_COLORS[method],
            label=method,
            lw=2.2 if emphasis else 1.3,
            linestyle=METHOD_DASHES[method],
            zorder=4 if emphasis else 3,
        )
        handles.append(line)
        axes[0].fill_between(
            steps,
            mean - ci,
            mean + ci,
            color=METHOD_COLORS[method],
            alpha=0.14 if emphasis else 0.07,
            lw=0,
            zorder=2,
        )

        regret_steps = sorted(
            step for name, step in regret_by_step if name == method
        )
        regret_mean = [
            float(np.mean(regret_by_step[(method, s)])) for s in regret_steps
        ]
        axes[1].plot(
            regret_steps,
            regret_mean,
            color=METHOD_COLORS[method],
            lw=2.2 if emphasis else 1.3,
            linestyle=METHOD_DASHES[method],
            zorder=4 if emphasis else 3,
        )

    axes[0].set(
        xlim=(0, 3 * PHASE_LENGTH),
        ylim=(0, 100),
        xlabel="online decisions",
        ylabel="cumulative accuracy (%)",
    )
    axes[0].set_title("Accuracy so far", loc="left", pad=18, fontweight="medium")
    _draw_phase_bands(axes[0], 101)

    axes[1].set(xlim=(0, 3 * PHASE_LENGTH), xlabel="online decisions")
    axes[1].set_ylabel("cumulative regret")
    axes[1].set_title("Regret so far", loc="left", pad=18, fontweight="medium")
    _draw_phase_bands(axes[1], axes[1].get_ylim()[1])

    for axis in axes:
        axis.grid(axis="y", zorder=1)
        axis.set_axisbelow(True)

    figure.legend(
        handles=handles,
        loc="lower center",
        ncol=len(ordered),
        bbox_to_anchor=(0.5, -0.03),
        handlelength=1.5,
        columnspacing=1.6,
        handletextpad=0.5,
    )
    figure.tight_layout(pad=0.6, rect=(0, 0.05, 1, 1))
    figure.subplots_adjust(wspace=0.22)
    figure.savefig(
        figure_dir / "bandit_learning_curves.png",
        dpi=220,
        bbox_inches="tight",
    )
    plt.close(figure)


def replot_from_outputs(output_dir: Path, figure_dir: Path) -> None:
    """Redraw the published figures from stored artifacts, without the model."""
    summary = json.loads(
        (output_dir / "summary.json").read_text()
    )["summary"]
    with (output_dir / "learning_curves.csv").open() as handle:
        curves = list(csv.DictReader(handle))
    write_figures(summary, curves, figure_dir)
