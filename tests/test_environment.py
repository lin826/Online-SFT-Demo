"""Causal invariants owned by the notification environment."""

from inspect import signature

import numpy as np

from online_sdft.config import ACTIONS, FEATURE_DIM, STREAM_LENGTH
from online_sdft.environment import (
    StudentObservation,
    factual_feedback,
    make_stream,
    oracle_utilities,
    student_observation,
    teacher_policy,
)


def test_stream_is_one_prequential_sequence():
    stream = make_stream(0)
    assert len(stream) == STREAM_LENGTH
    assert all(event.x.shape == (FEATURE_DIM,) for event in stream)
    assert [
        event.phase
        for event in stream[:1] + stream[80:81] + stream[160:161]
    ] == [0, 1, 2]


def test_archive_never_generates_notification_feedback():
    event = make_stream(0)[0]
    outcomes = {
        factual_feedback(
            event,
            ACTIONS.index("ARCHIVE"),
            np.random.default_rng(seed),
        )["outcome"]
        for seed in range(200)
    }
    assert outcomes <= {"ORGANIC_INBOX_OPEN", "NO_OBSERVATION"}


def test_teacher_returns_soft_distribution_from_factual_record():
    event = make_stream(1)[0]
    rng = np.random.default_rng(9)
    action = ACTIONS.index("LATER")
    feedback = factual_feedback(event, action, rng)
    distribution = teacher_policy(event, action, feedback, rng)
    assert distribution.shape == (len(ACTIONS),)
    assert np.isclose(distribution.sum(), 1.0)
    assert np.all(distribution > 0)


def test_oracle_is_an_evaluation_utility_vector_only():
    event = make_stream(2)[0]
    assert oracle_utilities(event).shape == (len(ACTIONS),)


def test_student_observation_has_no_privileged_state():
    event = make_stream(3)[0]
    observation = student_observation(event)
    assert isinstance(observation, StudentObservation)
    assert observation.features.shape == (FEATURE_DIM,)
    assert "category=" in observation.text
    assert all(
        field not in observation.text
        for field in (
            "busy",
            "incident_on_call",
            "leisure_social",
            "manager_focus",
        )
    )
    assert "z" not in signature(StudentObservation).parameters
    assert "busy" not in signature(StudentObservation).parameters
