import numpy as np
from inspect import signature

from bandit_experiment import (
    ACTIONS, FEATURE_DIM, MODEL_ID, STREAM_LENGTH, LiquidLLMPolicy, context_text,
    factual_feedback, make_stream, oracle_utilities, teacher_policy,
)


def test_stream_is_single_prequential_sequence():
    stream = make_stream(0)
    assert len(stream) == STREAM_LENGTH
    assert all(event.x.shape == (FEATURE_DIM,) for event in stream)
    assert [event.phase for event in stream[:1] + stream[80:81] + stream[160:161]] == [0, 1, 2]


def test_archive_feedback_has_no_notification_click():
    event = make_stream(0)[0]
    outcomes = {
        factual_feedback(event, ACTIONS.index("ARCHIVE"), np.random.default_rng(i))["outcome"]
        for i in range(200)
    }
    assert outcomes <= {"ORGANIC_INBOX_OPEN", "NO_OBSERVATION"}


def test_teacher_rollout_is_soft_and_uses_factual_record():
    event = make_stream(1)[0]
    rng = np.random.default_rng(9)
    action = ACTIONS.index("LATER")
    feedback = factual_feedback(event, action, rng)
    q = teacher_policy(event, action, feedback, rng)
    assert q.shape == (len(ACTIONS),)
    assert np.isclose(q.sum(), 1.0)
    assert np.all(q > 0)


def test_oracle_is_scoring_only_utility_vector():
    # The oracle API returns utilities and is never accepted by teacher_policy or
    # the policy update API as an argument.
    event = make_stream(2)[0]
    assert oracle_utilities(event).shape == (len(ACTIONS),)


def test_llm_context_excludes_privileged_post_decision_state():
    event = make_stream(3)[0]
    prompt_context = context_text(event)
    assert MODEL_ID == "LiquidAI/LFM2.5-230M"
    assert "category=" in prompt_context
    assert all(field not in prompt_context for field in (
        "busy", "incident_on_call", "leisure_social", "manager_focus"
    ))
    assert "event" not in signature(LiquidLLMPolicy.render_prompt).parameters
    assert "context" in signature(LiquidLLMPolicy.render_prompt).parameters
