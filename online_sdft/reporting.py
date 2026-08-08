"""Aggregation, qualitative selection, and plotting.

This module reads completed chronological traces. It never participates in an
action or update, which keeps evaluation-only oracle fields out of methods.
"""

from __future__ import annotations

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


def write_figures(
    summary: dict,
    curves: list[dict],
    figure_dir: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure_dir.mkdir(parents=True, exist_ok=True)
    colors = {
        "Base": "#9aa0a6",
        "ICL": "#e8710a",
        "RAG": "#d93025",
        "REINFORCE": "#00897b",
        "Online-SFT": "#7b3fa0",
        "Online-SDFT": "#1a73e8",
    }

    figure, axes = plt.subplots(1, 2, figsize=(11.8, 4.4))
    positions = np.arange(len(METHODS))
    accuracy = [
        100 * summary[method]["online_accuracy"]["mean"]
        for method in METHODS
    ]
    accuracy_error = [
        100 * summary[method]["online_accuracy"]["ci95"]
        for method in METHODS
    ]
    regret = [
        summary[method]["cum_regret"]["mean"]
        for method in METHODS
    ]
    regret_error = [
        summary[method]["cum_regret"]["ci95"]
        for method in METHODS
    ]
    for axis, values, errors, title, ylabel in (
        (
            axes[0],
            accuracy,
            accuracy_error,
            "Prequential online accuracy",
            "Online accuracy (%)",
        ),
        (
            axes[1],
            regret,
            regret_error,
            "Cumulative contextual-bandit regret",
            "Regret (lower is better)",
        ),
    ):
        bars = axis.bar(
            positions,
            values,
            yerr=errors,
            capsize=4,
            color=[colors[method] for method in METHODS],
            width=0.68,
        )
        axis.set_xticks(positions, METHODS, rotation=14, ha="right")
        axis.set_title(title, fontweight="bold")
        axis.set_ylabel(ylabel)
        axis.grid(axis="y", alpha=0.25)
        for bar, value in zip(bars, values):
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                value,
                f"{value:.1f}",
                ha="center",
                va="bottom",
                fontsize=8,
                fontweight="bold",
            )
    figure.suptitle(
        "Online notification routing · mean ± 95% CI",
        fontweight="bold",
    )
    figure.tight_layout()
    figure.savefig(
        figure_dir / "bandit_accuracy.png",
        dpi=170,
        bbox_inches="tight",
    )
    plt.close(figure)

    accuracy_by_step = defaultdict(list)
    for row in curves:
        accuracy_by_step[(row["method"], int(row["t"]))].append(
            float(row["cum_accuracy"])
        )

    figure, axes = plt.subplots(1, 2, figsize=(12.2, 4.5))
    for method in METHODS:
        steps = sorted(
            step
            for name, step in accuracy_by_step
            if name == method
        )
        values = np.array(
            [accuracy_by_step[(method, step)] for step in steps]
        )
        mean = np.array([step_values.mean() for step_values in values]) * 100
        ci = np.array(
            [
                0.0
                if len(step_values) == 1
                else 1.96
                * step_values.std(ddof=1)
                / math.sqrt(len(step_values))
                for step_values in values
            ]
        ) * 100
        axes[0].plot(
            steps,
            mean,
            color=colors[method],
            label=method,
            lw=2.6 if method == "Online-SDFT" else 1.5,
        )
        axes[0].fill_between(
            steps,
            mean - ci,
            mean + ci,
            color=colors[method],
            alpha=0.09,
        )
    for boundary in (PHASE_LENGTH, 2 * PHASE_LENGTH):
        axes[0].axvline(
            boundary,
            color="#5f6368",
            ls="--",
            lw=1,
        )
    axes[0].set(
        title="Cumulative online accuracy",
        xlabel="Online decisions",
        ylabel="Accuracy so far (%)",
        ylim=(0, 100),
    )
    axes[0].grid(alpha=0.25)
    axes[0].legend(fontsize=8, ncol=2)

    regret_by_step = defaultdict(list)
    for row in curves:
        regret_by_step[(row["method"], int(row["t"]))].append(
            float(row["cum_regret"])
        )
    for method in METHODS:
        steps = sorted(
            step
            for name, step in regret_by_step
            if name == method
        )
        mean = [
            np.mean(regret_by_step[(method, step)])
            for step in steps
        ]
        axes[1].plot(
            steps,
            mean,
            color=colors[method],
            label=method,
            lw=2.6 if method == "Online-SDFT" else 1.5,
        )
    for boundary in (PHASE_LENGTH, 2 * PHASE_LENGTH):
        axes[1].axvline(
            boundary,
            color="#5f6368",
            ls="--",
            lw=1,
        )
    axes[1].set(
        title="Accumulated regret",
        xlabel="Online decisions",
        ylabel="Cumulative regret",
    )
    axes[1].grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(
        figure_dir / "bandit_learning_curves.png",
        dpi=170,
        bbox_inches="tight",
    )
    plt.close(figure)
