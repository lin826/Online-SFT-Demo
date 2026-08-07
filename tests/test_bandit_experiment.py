import numpy as np

from bandit_experiment import (
    ACTIONS, FEATURE_DIM, STREAM_LENGTH, factual_feedback, make_stream,
    oracle_utilities, teacher_policy,
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
