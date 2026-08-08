"""Method-level tests independent of environment reward implementation."""

from inspect import getsource, signature

import numpy as np

import online_sdft.methods as methods_module
from online_sdft.config import CATEGORIES, ICL_K, METHODS, MODEL_ID
from online_sdft.environment import StudentObservation
from online_sdft.methods import (
    AGENT_CLASSES,
    ICLAgent,
    LiquidLLMPolicy,
    OnlineSDFTAgent,
    OnlineSFTAgent,
    RAGAgent,
    mixed_context_similarity,
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


def visible_observation(
    name,
    category=0,
    importance=0.5,
    deadline=0.5,
    affinity=0.5,
    hour=12.0,
    regime=0,
):
    angle = 2 * np.pi * hour / 24
    category_features = np.eye(len(CATEGORIES))[category]
    features = np.concatenate(
        [
            category_features,
            np.array(
                [
                    importance,
                    deadline,
                    affinity,
                    np.sin(angle),
                    np.cos(angle),
                    regime / 2,
                    1.0,
                    importance * deadline,
                ]
            ),
        ]
    )
    return StudentObservation(name, features)


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


def test_icl_uses_only_latest_teacher_examples_in_chronological_order():
    agent = ICLAgent(RecordingPolicy())
    for index in range(ICL_K + 2):
        agent.observe(
            visible_observation(f"past-{index}"),
            np.array([0.2, 0.5, 0.3]),
            teacher_action=index % 3,
            feedback={"outcome": "factual"},
            rng=np.random.default_rng(index),
        )
    examples = agent.prompt_examples(visible_observation("current"))
    assert [row["context"] for row in examples] == [
        f"past-{index}" for index in range(2, ICL_K + 2)
    ]
    assert all(row["context"] != "current" for row in examples)


def test_rag_retrieves_visible_nearest_neighbors_best_match_last(monkeypatch):
    monkeypatch.setattr(methods_module, "RAG_K", 2)
    agent = RAGAgent(RecordingPolicy())
    exact_old = visible_observation("exact-old", hour=23.5, regime=2)
    distractor = visible_observation(
        "distractor",
        category=1,
        importance=0.0,
        deadline=0.0,
        affinity=0.0,
        hour=12.0,
        regime=0,
    )
    exact_new = visible_observation("exact-new", hour=23.5, regime=2)
    for index, past in enumerate((exact_old, distractor, exact_new)):
        agent.observe(
            past,
            np.array([0.2, 0.5, 0.3]),
            teacher_action=index,
            feedback={"outcome": "factual"},
            rng=np.random.default_rng(index),
        )
    current = visible_observation("current", hour=23.5, regime=2)
    examples = agent.prompt_examples(current)
    assert [row["context"] for row in examples] == [
        "exact-old",
        "exact-new",
    ]


def test_mixed_similarity_handles_midnight_as_circular_time():
    before_midnight = visible_observation("before", hour=23.5)
    after_midnight = visible_observation("after", hour=0.5)
    noon = visible_observation("noon", hour=12.0)
    assert mixed_context_similarity(before_midnight, after_midnight) > (
        mixed_context_similarity(before_midnight, noon)
    )


def test_icl_prompt_uses_identical_demonstration_and_query_schema():
    class EchoTokenizer:
        @staticmethod
        def apply_chat_template(messages, **kwargs):
            del kwargs
            return messages[-1]["content"]

    policy = object.__new__(LiquidLLMPolicy)
    policy.tokenizer = EchoTokenizer()
    rendered = policy.render_prompt(
        "category=manager",
        [{"context": "category=calendar", "teacher_action": 1}],
    )
    assert "example 1 notification: category=calendar" in rendered
    assert "example 1 route: B" in rendered
    assert "current notification: category=manager" in rendered
    assert rendered.endswith("current route:")


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


def test_liquid_policy_uses_fp16_only_on_cuda_and_fp32_logits():
    source = getsource(LiquidLLMPolicy)
    assert 'self.device.type == "cuda"' in source
    assert "torch.float16" in source
    assert "else torch.float32" in source
    assert ".logits[:, -1, :].float()" in source
