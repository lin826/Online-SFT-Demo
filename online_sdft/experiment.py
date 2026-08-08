"""Prequential experiment loop and artifact orchestration.

This module is the only place where a method interacts with the environment.
The ordering is explicit: observe context, act, freeze the score, execute one
route, obtain teacher feedback, then update for the next round.
"""

from __future__ import annotations

import csv
import gc
import json
from collections import Counter
from pathlib import Path

import numpy as np

from .config import (
    ACTIONS,
    EXPLORATION_EPSILON,
    FIG,
    ICL_K,
    LORA_ALPHA,
    LORA_R,
    METHODS,
    MODEL_ID,
    ONLINE_BATCH_SIZE,
    OUT,
    PHASE_LENGTH,
    RAG_K,
    REINFORCE_BASELINE_STEP,
    REINFORCE_ENTROPY_COEF,
    REINFORCE_LR,
    REGIMES,
    REPLAY_SIZE,
    SDFT_LR,
    SFT_LR,
    STREAM_LENGTH,
    TEACHER_TEMPERATURE,
)
from .environment import (
    DEFAULT_ENVIRONMENT,
    Event,
    NotificationRoutingEnvironment,
    one_hot,
)
from .methods import LiquidLLMPolicy, StudentPolicy, create_agent
from .reporting import write_compact_results, write_figures


METHOD_RNG_OFFSETS = {
    "Base": 5,
    "ICL": 18,
    "RAG": 31,
    "Online-SFT": 44,
    "Online-SDFT": 57,
    "REINFORCE": 70,
}


def epsilon_greedy(probs: np.ndarray) -> np.ndarray:
    """Return the declared serving distribution over the student's ranking."""
    greedy = one_hot(int(np.argmax(probs)), len(ACTIONS))
    behavior = (
        (1 - EXPLORATION_EPSILON) * greedy
        + EXPLORATION_EPSILON / len(ACTIONS)
    )
    return behavior / behavior.sum()


def policy_sampling(probs: np.ndarray) -> np.ndarray:
    """Normalize the differentiable policy used by REINFORCE to act."""
    behavior = np.clip(np.asarray(probs, dtype=float), 1e-8, None)
    return behavior / behavior.sum()


def run_method(
    seed: int,
    method: str,
    stream: list[Event],
    policy: StudentPolicy,
    rollout_writer,
    curve_writer,
    environment: NotificationRoutingEnvironment = DEFAULT_ENVIRONMENT,
) -> dict:
    """Run one method on one stream with predict-then-learn ordering."""
    rng = np.random.default_rng(seed * 100 + METHOD_RNG_OFFSETS[method])
    agent = create_agent(method, policy)
    cumulative_regret = 0.0
    cumulative_correct = 0
    phase_correct = Counter()
    phase_total = Counter()
    phase_regret = Counter()

    for step, event in enumerate(stream, start=1):
        observation = environment.student_observation(event)
        student_probs = agent.action_probs(observation)
        behavior_probs = (
            policy_sampling(student_probs)
            if agent.samples_from_policy
            else epsilon_greedy(student_probs)
        )
        action = int(
            rng.choice(len(ACTIONS), p=behavior_probs)
        )

        # Freeze evaluation before any factual outcome or update exists.
        utilities = environment.oracle_utilities(event)
        oracle_action = int(np.argmax(utilities))
        step_regret = float(
            utilities[oracle_action] - utilities[action]
        )
        correct = int(action == oracle_action)
        cumulative_regret += step_regret
        cumulative_correct += correct
        phase_correct[event.phase] += correct
        phase_total[event.phase] += 1
        phase_regret[event.phase] += step_regret

        # Execute only the chosen route. REINFORCE consumes factual reward
        # directly; imitation methods receive a post-decision teacher target.
        feedback = environment.execute(event, action, rng)
        teacher_probs = None
        teacher_action = None
        if agent.uses_teacher:
            teacher_probs = environment.teacher_distribution(
                event,
                action,
                feedback,
                rng,
            )
            teacher_action = int(
                rng.choice(len(ACTIONS), p=teacher_probs)
            )

        agent.observe(
            observation,
            action,
            teacher_probs,
            teacher_action,
            feedback,
            rng,
        )

        record = {
            "seed": seed,
            "method": method,
            "t": step,
            "event_id": event.event_id,
            "phase": event.phase,
            "regime": REGIMES[event.phase],
            "category": event.category,
            "student_probs": dict(
                zip(ACTIONS, map(float, student_probs))
            ),
            "behavior_probs": dict(
                zip(ACTIONS, map(float, behavior_probs))
            ),
            "action": ACTIONS[action],
            "feedback": feedback,
            "teacher_probs": (
                None
                if teacher_probs is None
                else dict(zip(ACTIONS, map(float, teacher_probs)))
            ),
            "teacher_rollout": (
                None
                if teacher_action is None
                else ACTIONS[teacher_action]
            ),
            "oracle_action_scoring_only": ACTIONS[oracle_action],
            "correct_online": correct,
            "step_regret": step_regret,
            "cum_regret": cumulative_regret,
            "cum_accuracy": cumulative_correct / step,
        }
        rollout_writer.write(json.dumps(record) + "\n")

        curve_writer.writerow(
            {
                "seed": seed,
                "method": method,
                "t": step,
                "phase": event.phase,
                "regime": REGIMES[event.phase],
                "step_correct": correct,
                "step_regret": step_regret,
                "cum_accuracy": cumulative_correct / step,
                "cum_regret": cumulative_regret,
            }
        )

    return {
        "seed": seed,
        "method": method,
        "online_accuracy": cumulative_correct / len(stream),
        "cum_regret": cumulative_regret,
        "regret_per_decision": cumulative_regret / len(stream),
        **{
            f"online_accuracy_{REGIMES[phase]}": (
                phase_correct[phase] / phase_total[phase]
            )
            for phase in range(3)
        },
        **{
            f"regret_{REGIMES[phase]}": phase_regret[phase]
            for phase in range(3)
        },
    }


def experiment_config(
    seeds: int,
    seed_start: int,
    model_id: str,
    policy: LiquidLLMPolicy,
) -> dict:
    return {
        "seeds": seeds,
        "seed_start": seed_start,
        "stream_length": STREAM_LENGTH,
        "phase_length": PHASE_LENGTH,
        "actions": ACTIONS,
        "methods": METHODS,
        "student_model": model_id,
        "student_policy": "next-token A/B/C probabilities",
        "adapter": "LoRA",
        "lora_r": LORA_R,
        "lora_alpha": LORA_ALPHA,
        "trainable_parameters": policy.trainable_parameters,
        "device": str(policy.device),
        "exploration_epsilon": EXPLORATION_EPSILON,
        "behavior_policy": (
            "epsilon-greedy for non-RL methods; LFM policy sampling for "
            "REINFORCE"
        ),
        "replay_size": REPLAY_SIZE,
        "online_batch_size": ONLINE_BATCH_SIZE,
        "icl_examples": ICL_K,
        "rag_examples": RAG_K,
        "rag_similarity": (
            "equal-weight mixed visible fields; circular hour"
        ),
        "reinforce_lr": REINFORCE_LR,
        "reinforce_batch_size": 1,
        "reinforce_baseline": (
            f"causal reward EMA; step={REINFORCE_BASELINE_STEP}"
        ),
        "reinforce_entropy_coef": REINFORCE_ENTROPY_COEF,
        "sft_lr": SFT_LR,
        "sdft_lr": SDFT_LR,
        "teacher_temperature": TEACHER_TEMPERATURE,
        "evaluation": "prequential one-stream; predict then learn",
        "learning_signal": (
            "teacher targets for imitation; factual reward only for "
            "REINFORCE; oracle scoring only"
        ),
    }


def main(
    seeds: int = 3,
    model_id: str = MODEL_ID,
    device: str = "auto",
    local_files_only: bool = False,
    seed_start: int = 0,
    output_dir: Path | None = None,
    figure_dir: Path | None = None,
    environment: NotificationRoutingEnvironment = DEFAULT_ENVIRONMENT,
) -> None:
    """Run all methods and write raw plus compact experiment artifacts."""
    output_dir = output_dir or OUT
    figure_dir = figure_dir or FIG
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    print(f"loading Liquid student {model_id}", flush=True)

    rollouts_path = output_dir / "rollouts.jsonl"
    curves_path = output_dir / "learning_curves.csv"
    metrics_path = output_dir / "per_seed_metrics.csv"
    curve_fields = [
        "seed",
        "method",
        "t",
        "phase",
        "regime",
        "step_correct",
        "step_regret",
        "cum_accuracy",
        "cum_regret",
    ]
    metrics = []
    config = None

    with (
        rollouts_path.open("w") as rollout_file,
        curves_path.open("w", newline="") as curve_file,
    ):
        curve_writer = csv.DictWriter(
            curve_file,
            fieldnames=curve_fields,
            lineterminator="\n",
        )
        curve_writer.writeheader()
        for seed_index, seed in enumerate(
            range(seed_start, seed_start + seeds),
            start=1,
        ):
            policy = LiquidLLMPolicy(
                model_id=model_id,
                device=device,
                local_files_only=local_files_only,
            )
            if config is None:
                config = experiment_config(
                    seeds,
                    seed_start,
                    model_id,
                    policy,
                )
                print(
                    f"device={policy.device} "
                    f"trainable_lora={policy.trainable_parameters:,}",
                    flush=True,
                )
            stream = environment.make_stream(seed)
            for method in METHODS:
                metrics.append(
                    run_method(
                        seed,
                        method,
                        stream,
                        policy,
                        rollout_file,
                        curve_writer,
                        environment,
                    )
                )
                print(
                    f"seed {seed_index}/{seeds} "
                    f"(id={seed}) · {method}",
                    flush=True,
                )

            policy.optimizer = None
            del policy
            gc.collect()
            if config["device"] == "mps":
                import torch

                torch.mps.empty_cache()

    if not metrics or config is None:
        raise ValueError("seeds must be positive")

    with metrics_path.open("w", newline="") as metrics_file:
        writer = csv.DictWriter(
            metrics_file,
            fieldnames=list(metrics[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(metrics)

    with rollouts_path.open() as rollout_file:
        rollouts = [
            json.loads(line)
            for line in rollout_file
        ]
    with curves_path.open() as curve_file:
        curves = list(csv.DictReader(curve_file))

    summary = write_compact_results(
        output_dir,
        config,
        metrics,
        rollouts,
    )
    write_figures(summary, curves, figure_dir)
    print(f"wrote experiment artifacts to {output_dir}")
