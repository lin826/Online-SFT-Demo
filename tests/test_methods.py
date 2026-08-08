"""Method-level tests independent of environment reward implementation."""

from inspect import getsource, signature

import numpy as np

import online_sdft.methods as methods_module
from online_sdft.config import METHODS, MODEL_ID
from online_sdft.environment import StudentObservation
from online_sdft.methods import (
    AGENT_CLASSES,
    LiquidLLMPolicy,
    OnlineSDFTAgent,
    OnlineSFTAgent,
)


class RecordingPolicy:
    def __init__(self):
        self.learning_rate = None
        self.updates = []

    def start_run(self, learning_rate):
        self.learning_rate = learning_rate
        self.updates.clear()

    def probs(self, context, examples=None):
        del context, examples
        return np.array([0.2, 0.5, 0.3])

    def update(self, batch):
        self.updates.append(batch)
        return 0.0


def test_method_registry_matches_reported_benchmark():
    assert tuple(AGENT_CLASSES) == METHODS


def test_online_sft_keeps_one_hard_teacher_draw():
    policy = RecordingPolicy()
    agent = OnlineSFTAgent(policy)
    observation = StudentObservation("visible", np.ones(3))
    agent.observe(
        observation,
        np.array([0.1, 0.7, 0.2]),
        teacher_action=1,
        feedback={"outcome": "factual"},
        rng=np.random.default_rng(0),
    )
    target = policy.updates[0][0][1]
    assert np.array_equal(target, np.array([0.0, 1.0, 0.0]))


def test_online_sdft_keeps_full_soft_teacher_distribution():
    policy = RecordingPolicy()
    agent = OnlineSDFTAgent(policy)
    observation = StudentObservation("visible", np.ones(3))
    distribution = np.array([0.1, 0.7, 0.2])
    agent.observe(
        observation,
        distribution,
        teacher_action=1,
        feedback={"outcome": "factual"},
        rng=np.random.default_rng(0),
    )
    target = policy.updates[0][0][1]
    assert np.array_equal(target, distribution)
    assert target is not distribution


def test_llm_api_accepts_visible_text_not_full_event():
    assert MODEL_ID == "LiquidAI/LFM2.5-230M"
    parameters = signature(LiquidLLMPolicy.render_prompt).parameters
    assert "event" not in parameters
    assert "context" in parameters


def test_methods_module_has_no_reward_or_oracle_implementation():
    source = getsource(methods_module)
    assert "oracle_utilities" not in source
    assert "factual_feedback" not in source
    assert "teacher_distribution(" not in source
