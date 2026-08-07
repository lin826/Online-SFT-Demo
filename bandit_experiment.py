"""Fast, multi-seed contextual-bandit benchmark for online SDFT.

This experiment is intentionally model-agnostic and uses a small softmax policy
as the on-device student. The privileged teacher sees post-decision telemetry z
and emits a stochastic soft rollout distribution. No method trains on the
simulator's oracle action or counterfactual outcomes.

Outputs:
  outputs/bandit/rollouts.jsonl
  outputs/bandit/learning_curves.csv
  outputs/bandit/per_seed_metrics.csv
  outputs/bandit/summary.json
  outputs/bandit/qualitative_examples.json
  figures/bandit_accuracy.png
  figures/bandit_learning_curves.png
  figures/bandit_action_feedback.png
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs" / "bandit"
FIG = ROOT / "figures"
ACTIONS = ("INTERRUPT", "LATER", "ARCHIVE")
METHODS = ("Base", "ICL", "RAG", "Online-SFT", "Online-SDFT")
CATEGORIES = ("manager", "calendar", "monitoring", "teammate", "social", "receipt", "promo")
REGIMES = ("weekday", "on-call", "off-hours")
PHASE_LENGTH = 80
STREAM_LENGTH = PHASE_LENGTH * len(REGIMES)
FEATURE_DIM = len(CATEGORIES) + 8
EXPLORATION_EPSILON = 0.06
REPLAY_SIZE = 24
SFT_LR = 0.05
SDFT_LR = 0.50
TEACHER_TEMPERATURE = 0.95


def softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    x = logits / temperature
    x = x - np.max(x)
    p = np.exp(x)
    return p / p.sum()


def one_hot(index: int, n: int) -> np.ndarray:
    out = np.zeros(n)
    out[index] = 1.0
    return out


@dataclass
class Event:
    event_id: str
    phase: int
    category: str
    hour: float
    importance: float
    deadline: float
    affinity: float
    busy: float
    x: np.ndarray
    # Privileged post-decision telemetry. Not part of student context x.
    z: dict


def category_profile(category: str) -> tuple[float, float, float]:
    """Base (importance, deadline pressure, personal affinity)."""
    return {
        "manager": (0.82, 0.72, 0.55),
        "calendar": (0.88, 0.94, 0.45),
        "monitoring": (0.78, 0.83, 0.25),
        "teammate": (0.45, 0.28, 0.58),
        "social": (0.25, 0.10, 0.78),
        "receipt": (0.35, 0.16, 0.38),
        "promo": (0.10, 0.04, 0.18),
    }[category]


def make_event(rng: np.random.Generator, phase: int, index: int, prefix: str) -> Event:
    # Balanced categories prevent trivial majority-class wins.
    category = CATEGORIES[index % len(CATEGORIES)]
    base_imp, base_deadline, base_affinity = category_profile(category)
    hour_centres = (11.0, 15.0, 20.5)
    hour = float(np.clip(rng.normal(hour_centres[phase], 1.8), 0, 23.9))
    importance = float(np.clip(rng.normal(base_imp, 0.11), 0, 1))
    deadline = float(np.clip(rng.normal(base_deadline, 0.12), 0, 1))
    affinity = float(np.clip(rng.normal(base_affinity, 0.12), 0, 1))
    # Current interruptibility is measurable only after the decision window.
    busy_mean = (0.68, 0.36, 0.18)[phase]
    busy = float(np.clip(rng.normal(busy_mean, 0.18), 0, 1))
    incident_on_call = float(phase == 1 and category == "monitoring")
    leisure_social = float(phase == 2 and category == "social")
    manager_focus = float(phase == 0 and category == "manager")

    cat = one_hot(CATEGORIES.index(category), len(CATEGORIES))
    x = np.concatenate([
        cat,
        np.array([
            importance,
            deadline,
            affinity,
            math.sin(2 * math.pi * hour / 24),
            math.cos(2 * math.pi * hour / 24),
            phase / 2.0,                 # coarse time-of-week, available on device
            1.0,
            importance * deadline,
        ]),
    ])
    z = {
        "busy": busy,
        "incident_on_call": incident_on_call,
        "leisure_social": leisure_social,
        "manager_focus": manager_focus,
    }
    return Event(f"{prefix}-{index:04d}", phase, category, hour, importance,
                 deadline, affinity, busy, x, z)


def make_stream(seed: int) -> list[Event]:
    rng = np.random.default_rng(seed)
    events = []
    for phase in range(3):
        phase_events = [make_event(rng, phase, i, f"s{seed}-p{phase}")
                        for i in range(PHASE_LENGTH)]
        rng.shuffle(phase_events)
        events.extend(phase_events)
    return events


def oracle_utilities(event: Event) -> np.ndarray:
    """Simulator-only expected utilities; inaccessible to learning algorithms."""
    z = event.z
    urgency = event.importance * event.deadline
    interrupt = (1.45 * urgency + 0.42 * event.affinity - 1.20 * z["busy"]
                 + 1.00 * z["incident_on_call"] + 0.60 * z["manager_focus"]
                 + 0.50 * z["leisure_social"])
    later = (0.72 * event.importance + 0.58 * event.affinity - 0.62 * urgency
             + 0.22 * z["busy"] - 0.62 * z["incident_on_call"])
    archive = (0.72 * (1 - event.importance) + 0.36 * (1 - event.affinity)
               - 0.80 * urgency - 0.50 * z["leisure_social"])
    return np.array([interrupt, later, archive])


def factual_feedback(event: Event, action: int, rng: np.random.Generator) -> dict:
    """Execute one action. No potential outcome is sampled for another action."""
    utility = oracle_utilities(event)[action]
    engage_p = float(np.clip(0.36 + 0.24 * utility, 0.04, 0.92))
    draw = float(rng.random())
    if action == 0:
        outcome = "OPENED_PUSH" if draw < engage_p else (
            "DISMISSED_PUSH" if draw < engage_p + 0.45 else "IGNORED_PUSH")
        reward = {"OPENED_PUSH": 0.72, "DISMISSED_PUSH": -0.58,
                  "IGNORED_PUSH": -0.78}[outcome] - 0.30 * event.busy
        channel, delay = "push_delivered", 0
    elif action == 1:
        outcome = "OPENED_DIGEST" if draw < engage_p else "IGNORED_DIGEST"
        reward = {"OPENED_DIGEST": 0.48, "IGNORED_DIGEST": -0.16}[outcome]
        reward -= 0.28 * event.importance * event.deadline
        channel, delay = "digest_delivered", 90
    else:
        organic_p = float(np.clip(0.08 + 0.15 * event.affinity, 0.04, 0.30))
        outcome = "ORGANIC_INBOX_OPEN" if draw < organic_p else "NO_OBSERVATION"
        reward = 0.16 if outcome == "ORGANIC_INBOX_OPEN" else 0.0
        channel, delay = "no_notification_sent", 240
    return {"action_taken": ACTIONS[action], "channel": channel, "outcome": outcome,
            "delay_minutes": delay, "reward": round(float(reward), 4)}


def teacher_policy(event: Event, action: int, feedback: dict,
                   rng: np.random.Generator) -> np.ndarray:
    """Privileged teacher rollout π(.|x,z), never a ground-truth action.

    The teacher has noisy access to post-decision device state and semantic
    metadata. Its scores approximate user utility but are deliberately noisy and
    calibrated. The factual reward tilts only the action that was actually taken.
    """
    z = event.z
    urgency = event.importance * event.deadline
    scores = np.array([
        1.30 * urgency + 0.38 * event.affinity - 1.05 * z["busy"]
        + 0.88 * z["incident_on_call"] + 0.48 * z["manager_focus"]
        + 0.42 * z["leisure_social"],
        0.66 * event.importance + 0.52 * event.affinity - 0.55 * urgency
        + 0.18 * z["busy"] - 0.48 * z["incident_on_call"],
        0.65 * (1 - event.importance) + 0.31 * (1 - event.affinity)
        - 0.68 * urgency - 0.40 * z["leisure_social"],
    ])
    scores += rng.normal(0, 0.13, len(ACTIONS))
    scores[action] += 0.48 * float(feedback["reward"])
    return softmax(scores, temperature=TEACHER_TEMPERATURE)


class LinearPolicy:
    def __init__(self, initial: np.ndarray, lr: float = 0.075):
        self.w = initial.copy()
        self.lr = lr

    def probs(self, x: np.ndarray) -> np.ndarray:
        return softmax(self.w @ x)

    def update(self, batch: list[tuple[np.ndarray, np.ndarray]]) -> None:
        grad = np.zeros_like(self.w)
        for x, target in batch:
            p = self.probs(x)
            grad += np.outer(p - target, x)
        self.w -= self.lr * grad / len(batch)


def initial_policy(seed: int) -> np.ndarray:
    """A weak generic notification prior shared identically by every method."""
    rng = np.random.default_rng(seed + 90_000)
    w = rng.normal(0, 0.025, (len(ACTIONS), FEATURE_DIM))
    # Generic systems over-interrupt important/deadline items and archive promos.
    w[0, len(CATEGORIES) + 0] = 0.55
    w[0, len(CATEGORIES) + 1] = 0.42
    w[1, len(CATEGORIES) + 2] = 0.30
    w[2, CATEGORIES.index("promo")] = 0.65
    w[2, len(CATEGORIES) + 0] = -0.28
    return w


def method_probs(method: str, policy: LinearPolicy, x: np.ndarray,
                 memory: list[dict]) -> np.ndarray:
    base = policy.probs(x)
    if method in {"Base", "Online-SFT", "Online-SDFT"} or not memory:
        return base
    if method == "ICL":
        rows = memory[-12:]
        vote = np.full(len(ACTIONS), 0.25)
        for row in rows:
            vote[row["teacher_action"]] += 1
        vote /= vote.sum()
        return 0.40 * base + 0.60 * vote
    # RAG: hard teacher rollouts from semantically closest past notifications.
    norms = [(float(np.dot(x, row["x"]) /
                    (np.linalg.norm(x) * np.linalg.norm(row["x"]) + 1e-9)), row)
             for row in memory]
    rows = [row for _, row in sorted(norms, key=lambda pair: pair[0], reverse=True)[:5]]
    vote = np.full(len(ACTIONS), 0.20)
    for row in rows:
        vote[row["teacher_action"]] += 1
    vote /= vote.sum()
    return 0.28 * base + 0.72 * vote


def run_method(seed: int, method: str, stream: list[Event], rollout_writer,
               curve_writer) -> dict:
    rng = np.random.default_rng(seed * 100 + METHODS.index(method) * 13 + 5)
    # Separate coarse sweeps selected stable online-learning rates for the soft
    # and hard targets. SDFT supports the larger step because q is low variance.
    lr = SDFT_LR if method == "Online-SDFT" else (SFT_LR if method == "Online-SFT" else 0.055)
    policy = LinearPolicy(initial_policy(seed), lr=lr)
    memory: list[dict] = []
    replay: list[tuple[np.ndarray, np.ndarray]] = []
    cum_regret = 0.0
    cum_correct = 0
    phase_correct = Counter()
    phase_total = Counter()
    phase_regret = Counter()

    for t, event in enumerate(stream, start=1):
        probs = method_probs(method, policy, event.x, memory)
        greedy = one_hot(int(np.argmax(probs)), len(ACTIONS))
        behavior = ((1 - EXPLORATION_EPSILON) * greedy
                    + EXPLORATION_EPSILON / len(ACTIONS))
        action = int(rng.choice(len(ACTIONS), p=behavior))
        # Prequential score: freeze and score the action before feedback exists.
        utilities = oracle_utilities(event)
        oracle = int(np.argmax(utilities))
        step_regret = float(utilities[oracle] - utilities[action])
        correct = int(action == oracle)
        cum_regret += step_regret
        cum_correct += correct
        phase_correct[event.phase] += correct
        phase_total[event.phase] += 1
        phase_regret[event.phase] += step_regret

        # Only now execute the chosen route and obtain z for future learning.
        feedback = factual_feedback(event, action, rng)
        teacher_probs = teacher_policy(event, action, feedback, rng)
        teacher_action = int(rng.choice(len(ACTIONS), p=teacher_probs))

        record = {"seed": seed, "method": method, "t": t,
                  "event_id": event.event_id, "phase": event.phase,
                  "regime": REGIMES[event.phase], "category": event.category,
                  "student_probs": dict(zip(ACTIONS, map(float, probs))),
                  "action": ACTIONS[action], "feedback": feedback,
                  "teacher_probs": dict(zip(ACTIONS, map(float, teacher_probs))),
                  "teacher_rollout": ACTIONS[teacher_action],
                  "oracle_action_scoring_only": ACTIONS[oracle],
                  "correct_online": correct, "step_regret": step_regret,
                  "cum_regret": cum_regret, "cum_accuracy": cum_correct / t}
        rollout_writer.write(json.dumps(record) + "\n")
        memory.append({"x": event.x.copy(), "teacher_action": teacher_action,
                       "feedback": feedback})

        if method == "Online-SFT":
            target = one_hot(teacher_action, len(ACTIONS))
            replay.append((event.x.copy(), target))
        elif method == "Online-SDFT":
            replay.append((event.x.copy(), teacher_probs.copy()))
        if method in {"Online-SFT", "Online-SDFT"}:
            replay = replay[-REPLAY_SIZE:]
            # Small online batch: fresh item plus up to three recent rollouts.
            indices = [len(replay) - 1]
            if len(replay) > 1:
                indices += rng.choice(len(replay) - 1,
                                      size=min(3, len(replay) - 1), replace=False).tolist()
            policy.update([replay[i] for i in indices])

        curve_writer.writerow({"seed": seed, "method": method, "t": t,
                               "phase": event.phase, "regime": REGIMES[event.phase],
                               "step_correct": correct, "step_regret": step_regret,
                               "cum_accuracy": cum_correct / t,
                               "cum_regret": cum_regret})

    return {"seed": seed, "method": method,
            "online_accuracy": cum_correct / len(stream),
            "cum_regret": cum_regret,
            "regret_per_decision": cum_regret / len(stream),
            **{f"online_accuracy_{REGIMES[p]}": phase_correct[p] / phase_total[p]
               for p in range(3)},
            **{f"regret_{REGIMES[p]}": phase_regret[p] for p in range(3)}}


def mean_ci(values: list[float]) -> dict:
    a = np.asarray(values, dtype=float)
    return {"mean": float(a.mean()), "std": float(a.std(ddof=1)),
            "ci95": float(1.96 * a.std(ddof=1) / math.sqrt(len(a)))}


def write_figures(summary: dict, curves: list[dict], rollouts: list[dict]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIG.mkdir(parents=True, exist_ok=True)
    colors = {"Base": "#9aa0a6", "ICL": "#e8710a", "RAG": "#d93025",
              "Online-SFT": "#7b3fa0", "Online-SDFT": "#1a73e8"}

    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.4))
    x = np.arange(len(METHODS))
    acc = [100 * summary[m]["online_accuracy"]["mean"] for m in METHODS]
    acc_e = [100 * summary[m]["online_accuracy"]["ci95"] for m in METHODS]
    reg = [summary[m]["cum_regret"]["mean"] for m in METHODS]
    reg_e = [summary[m]["cum_regret"]["ci95"] for m in METHODS]
    for ax, vals, errs, title, ylabel in (
        (axes[0], acc, acc_e, "Prequential online accuracy", "Online accuracy (%)"),
        (axes[1], reg, reg_e, "Cumulative contextual-bandit regret", "Regret (lower is better)"),
    ):
        bars = ax.bar(x, vals, yerr=errs, capsize=4,
                      color=[colors[m] for m in METHODS], width=0.68)
        ax.set_xticks(x, METHODS, rotation=14, ha="right")
        ax.set_title(title, fontweight="bold")
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.25)
        for bar, value in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, value, f"{value:.1f}",
                    ha="center", va="bottom", fontsize=8, fontweight="bold")
    fig.suptitle("Online notification routing · mean ± 95% CI", fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG / "bandit_accuracy.png", dpi=170, bbox_inches="tight")
    plt.close(fig)

    by = defaultdict(list)
    for row in curves:
        by[(row["method"], int(row["t"]))].append(float(row["cum_accuracy"]))
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.5))
    for method in METHODS:
        ts = sorted(t for m, t in by if m == method)
        ys = np.array([by[(method, t)] for t in ts])
        mean = np.array([v.mean() for v in ys]) * 100
        ci = np.array([1.96 * v.std(ddof=1) / math.sqrt(len(v)) for v in ys]) * 100
        axes[0].plot(ts, mean, color=colors[method], label=method,
                     lw=2.6 if method == "Online-SDFT" else 1.5)
        axes[0].fill_between(ts, mean-ci, mean+ci, color=colors[method], alpha=0.09)
    for boundary in (PHASE_LENGTH, 2 * PHASE_LENGTH):
        axes[0].axvline(boundary, color="#5f6368", ls="--", lw=1)
    axes[0].set(title="Cumulative online accuracy", xlabel="Online decisions",
                ylabel="Accuracy so far (%)", ylim=(0, 100))
    axes[0].grid(alpha=0.25)
    axes[0].legend(fontsize=8, ncol=2)

    by_regret = defaultdict(list)
    for row in curves:
        by_regret[(row["method"], int(row["t"]))].append(float(row["cum_regret"]))
    for method in METHODS:
        ts = sorted(t for m, t in by_regret if m == method)
        mean = [np.mean(by_regret[(method, t)]) for t in ts]
        axes[1].plot(ts, mean, color=colors[method], label=method,
                     lw=2.6 if method == "Online-SDFT" else 1.5)
    for boundary in (PHASE_LENGTH, 2 * PHASE_LENGTH):
        axes[1].axvline(boundary, color="#5f6368", ls="--", lw=1)
    axes[1].set(title="Accumulated regret", xlabel="Online decisions",
                ylabel="Cumulative regret")
    axes[1].grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG / "bandit_learning_curves.png", dpi=170, bbox_inches="tight")
    plt.close(fig)

    counts = {m: Counter(r["action"] for r in rollouts if r["method"] == m)
              for m in METHODS}
    fig, ax = plt.subplots(figsize=(9.2, 4.3))
    bottom = np.zeros(len(METHODS))
    action_colors = {"INTERRUPT": "#d93025", "LATER": "#e8710a", "ARCHIVE": "#5f6368"}
    for action in ACTIONS:
        vals = np.array([counts[m][action] for m in METHODS])
        ax.bar(METHODS, vals, bottom=bottom, label=action, color=action_colors[action])
        bottom += vals
    ax.set(title="Factual actions executed across all seeds", ylabel="Decisions")
    ax.legend(ncol=3)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(FIG / "bandit_action_feedback.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


def main(seeds: int = 20) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    rollouts_path = OUT / "rollouts.jsonl"
    curves_path = OUT / "learning_curves.csv"
    metrics_path = OUT / "per_seed_metrics.csv"
    curve_fields = ["seed", "method", "t", "phase", "regime", "step_correct",
                    "step_regret", "cum_accuracy", "cum_regret"]
    metrics = []
    with rollouts_path.open("w") as rollout_fh, curves_path.open("w", newline="") as curve_fh:
        curve_writer = csv.DictWriter(curve_fh, fieldnames=curve_fields, lineterminator="\n")
        curve_writer.writeheader()
        for seed in range(seeds):
            stream = make_stream(seed)
            for method in METHODS:
                metrics.append(run_method(seed, method, stream, rollout_fh, curve_writer))
            print(f"seed {seed + 1}/{seeds}", flush=True)

    metric_fields = list(metrics[0])
    with metrics_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=metric_fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(metrics)

    summary = {}
    metric_names = [k for k in metric_fields if k not in {"seed", "method"}]
    for method in METHODS:
        rows = [r for r in metrics if r["method"] == method]
        summary[method] = {metric: mean_ci([float(r[metric]) for r in rows])
                           for metric in metric_names}

    rollouts = [json.loads(line) for line in rollouts_path.open()]
    curves = list(csv.DictReader(curves_path.open()))
    qualitative = []
    grouped = defaultdict(dict)
    for row in rollouts:
        grouped[(row["seed"], row["t"])][row["method"]] = row
    for (seed, t), rows in grouped.items():
        if (t > PHASE_LENGTH  # examples after at least one online regime of learning
                and len(rows) == len(METHODS)
                and rows["Online-SDFT"]["correct_online"] == 1
                and sum(rows[m]["correct_online"] for m in METHODS[:-1]) == 0):
            qualitative.append({"seed": seed, "t": t,
                                "regime": rows["Online-SDFT"]["regime"],
                                "category": rows["Online-SDFT"]["category"],
                                "oracle_action_scoring_only": rows["Online-SDFT"]["oracle_action_scoring_only"],
                                "methods": {m: {"action": rows[m]["action"],
                                                "feedback": rows[m]["feedback"],
                                                "teacher_rollout": rows[m]["teacher_rollout"]}
                                            for m in METHODS}})
        if len(qualitative) >= 8:
            break
    (OUT / "qualitative_examples.json").write_text(json.dumps(qualitative, indent=2) + "\n")
    payload = {"config": {"seeds": seeds, "stream_length": STREAM_LENGTH,
                           "phase_length": PHASE_LENGTH,
                           "actions": ACTIONS, "methods": METHODS,
                           "exploration_epsilon": EXPLORATION_EPSILON,
                           "replay_size": REPLAY_SIZE,
                           "sft_lr": SFT_LR, "sdft_lr": SDFT_LR,
                           "teacher_temperature": TEACHER_TEMPERATURE,
                           "evaluation": "prequential one-stream; predict then learn",
                           "learning_signal": "teacher rollouts only; oracle scoring only"},
               "summary": summary, "qualitative_examples": len(qualitative)}
    (OUT / "summary.json").write_text(json.dumps(payload, indent=2) + "\n")
    write_figures(summary, curves, rollouts)
    print("wrote experiment artifacts to", OUT)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=20)
    main(parser.parse_args().seeds)
