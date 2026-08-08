"""Interaction-order tests for the experiment bridge."""

import csv
import io
import json

import numpy as np

from online_sdft.config import METHODS, STREAM_LENGTH
from online_sdft.environment import NotificationRoutingEnvironment
from online_sdft.experiment import run_method


class OrderedPolicy:
    def __init__(self, calls):
        self.calls = calls
        self.updates = []
        self.example_counts = []

    def start_run(self, learning_rate):
        self.learning_rate = learning_rate
        self.updates.clear()

    def probs(self, context, examples=None):
        del context
        self.calls.append("act")
        self.example_counts.append(len(examples or []))
        return np.array([0.2, 0.6, 0.2])

    def update(self, batch):
        self.calls.append("update")
        self.updates.append(batch)
        return 0.0


class OrderedEnvironment(NotificationRoutingEnvironment):
    def __init__(self, calls):
        self.calls = calls
        self.executing = False

    def student_observation(self, event):
        self.calls.append("context")
        return super().student_observation(event)

    def oracle_utilities(self, event):
        if not self.executing:
            self.calls.append("score")
        return super().oracle_utilities(event)

    def execute(self, event, action, rng):
        self.calls.append("execute")
        self.executing = True
        try:
            return super().execute(event, action, rng)
        finally:
            self.executing = False

    def teacher_distribution(self, event, action, feedback, rng):
        self.calls.append("teacher")
        return super().teacher_distribution(
            event,
            action,
            feedback,
            rng,
        )


def run_fast_method(method):
    calls = []
    environment = OrderedEnvironment(calls)
    stream = environment.make_stream(0)
    policy = OrderedPolicy(calls)
    rollout_buffer = io.StringIO()
    curve_buffer = io.StringIO()
    fields = [
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
    curve_writer = csv.DictWriter(
        curve_buffer,
        fieldnames=fields,
        lineterminator="\n",
    )
    curve_writer.writeheader()
    metrics = run_method(
        0,
        method,
        stream,
        policy,
        rollout_buffer,
        curve_writer,
        environment,
    )
    rollouts = [
        json.loads(line)
        for line in rollout_buffer.getvalue().splitlines()
    ]
    return calls, policy, metrics, rollouts


def test_predict_score_execute_teacher_update_order():
    calls, policy, metrics, rollouts = run_fast_method("Online-SDFT")
    assert calls[:6] == [
        "context",
        "act",
        "score",
        "execute",
        "teacher",
        "update",
    ]
    assert len(policy.updates) == STREAM_LENGTH
    assert len(rollouts) == STREAM_LENGTH
    assert metrics["method"] == "Online-SDFT"


def test_frozen_baselines_never_update_weights():
    for method in METHODS[:3]:
        _, policy, _, _ = run_fast_method(method)
        assert not policy.updates


def test_icl_and_rag_prompts_contain_strictly_past_records_only():
    for method in ("ICL", "RAG"):
        _, policy, _, _ = run_fast_method(method)
        assert policy.example_counts[:3] == [0, 1, 2]
        assert all(
            count <= step - 1
            for step, count in enumerate(
                policy.example_counts,
                start=1,
            )
        )
