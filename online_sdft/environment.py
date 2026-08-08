"""Notification-routing environment, rewards, and privileged views.

The environment owns the causal world: stream generation, action-dependent
feedback, and evaluator-only utilities. It exposes two disjoint projections of
:class:`Event` so that neither the deployed methods nor the teacher can read
state they are not entitled to.

:class:`StudentObservation` is the pre-decision view. :class:`TeacherObservation`
is the post-decision view: the same context plus the executed route, its
realized outcome, and the latent interruptibility state. Neither ever contains
:meth:`NotificationRoutingEnvironment.oracle_utilities`, which exists only to
score the benchmark.

The teacher itself is *not* defined here. It is a language model
(:class:`online_sdft.methods.LMTeacher`) that reads a
:class:`TeacherObservation`, so that no hand-written scoring function can
encode the evaluator's preferences.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .config import (
    ACTIONS,
    CATEGORIES,
    PHASE_LENGTH,
    REGIMES,
)


def softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    shifted = logits / temperature
    shifted = shifted - np.max(shifted)
    probabilities = np.exp(shifted)
    return probabilities / probabilities.sum()


def one_hot(index: int, size: int) -> np.ndarray:
    values = np.zeros(size)
    values[index] = 1.0
    return values


@dataclass
class Event:
    """Full simulator event, including evaluator/teacher-only state."""

    event_id: str
    phase: int
    category: str
    hour: float
    importance: float
    deadline: float
    affinity: float
    busy: float
    x: np.ndarray
    z: dict


@dataclass(frozen=True)
class StudentObservation:
    """The complete and only view supplied to a deployed method."""

    text: str
    features: np.ndarray


class NotificationRoutingEnvironment:
    """Causal contextual-bandit simulator for one notification stream."""

    @staticmethod
    def category_profile(category: str) -> tuple[float, float, float]:
        """Return base importance, deadline pressure, and affinity."""
        return {
            "manager": (0.82, 0.72, 0.55),
            "calendar": (0.88, 0.94, 0.45),
            "monitoring": (0.78, 0.83, 0.25),
            "teammate": (0.45, 0.28, 0.58),
            "social": (0.25, 0.10, 0.78),
            "receipt": (0.35, 0.16, 0.38),
            "promo": (0.10, 0.04, 0.18),
        }[category]

    def make_event(
        self,
        rng: np.random.Generator,
        phase: int,
        index: int,
        prefix: str,
    ) -> Event:
        # Balanced categories prevent trivial majority-class wins.
        category = CATEGORIES[index % len(CATEGORIES)]
        base_imp, base_deadline, base_affinity = self.category_profile(category)
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

        category_features = one_hot(CATEGORIES.index(category), len(CATEGORIES))
        features = np.concatenate(
            [
                category_features,
                np.array(
                    [
                        importance,
                        deadline,
                        affinity,
                        math.sin(2 * math.pi * hour / 24),
                        math.cos(2 * math.pi * hour / 24),
                        phase / 2.0,
                        1.0,
                        importance * deadline,
                    ]
                ),
            ]
        )
        privileged = {
            "busy": busy,
            "incident_on_call": incident_on_call,
            "leisure_social": leisure_social,
            "manager_focus": manager_focus,
        }
        return Event(
            f"{prefix}-{index:04d}",
            phase,
            category,
            hour,
            importance,
            deadline,
            affinity,
            busy,
            features,
            privileged,
        )

    def make_stream(self, seed: int) -> list[Event]:
        rng = np.random.default_rng(seed)
        events = []
        for phase in range(3):
            phase_events = [
                self.make_event(rng, phase, index, f"s{seed}-p{phase}")
                for index in range(PHASE_LENGTH)
            ]
            rng.shuffle(phase_events)
            events.extend(phase_events)
        return events

    @staticmethod
    def student_observation(event: Event) -> StudentObservation:
        """Project a full event onto student-visible information only."""
        text = (
            f"category={event.category}; hour={event.hour:.1f}; "
            f"regime={REGIMES[event.phase]}; importance={event.importance:.2f}; "
            f"deadline={event.deadline:.2f}; affinity={event.affinity:.2f}"
        )
        return StudentObservation(text=text, features=event.x.copy())

    @staticmethod
    def oracle_utilities(event: Event) -> np.ndarray:
        """Return evaluator-only utility; never a method training target."""
        z = event.z
        urgency = event.importance * event.deadline
        interrupt = (
            1.45 * urgency
            + 0.42 * event.affinity
            - 1.20 * z["busy"]
            + 1.00 * z["incident_on_call"]
            + 0.60 * z["manager_focus"]
            + 0.50 * z["leisure_social"]
        )
        later = (
            0.72 * event.importance
            + 0.58 * event.affinity
            - 0.62 * urgency
            + 0.22 * z["busy"]
            - 0.62 * z["incident_on_call"]
        )
        archive = (
            0.72 * (1 - event.importance)
            + 0.36 * (1 - event.affinity)
            - 0.80 * urgency
            - 0.50 * z["leisure_social"]
        )
        return np.array([interrupt, later, archive])

    def execute(
        self,
        event: Event,
        action: int,
        rng: np.random.Generator,
    ) -> dict:
        """Execute one action; never sample an unchosen potential outcome."""
        utility = self.oracle_utilities(event)[action]
        engage_probability = float(np.clip(0.36 + 0.24 * utility, 0.04, 0.92))
        draw = float(rng.random())
        if action == 0:
            outcome = (
                "OPENED_PUSH"
                if draw < engage_probability
                else (
                    "DISMISSED_PUSH"
                    if draw < engage_probability + 0.45
                    else "IGNORED_PUSH"
                )
            )
            reward = {
                "OPENED_PUSH": 0.72,
                "DISMISSED_PUSH": -0.58,
                "IGNORED_PUSH": -0.78,
            }[outcome] - 0.30 * event.busy
            channel, delay = "push_delivered", 0
        elif action == 1:
            outcome = (
                "OPENED_DIGEST"
                if draw < engage_probability
                else "IGNORED_DIGEST"
            )
            reward = {"OPENED_DIGEST": 0.48, "IGNORED_DIGEST": -0.16}[outcome]
            reward -= 0.28 * event.importance * event.deadline
            channel, delay = "digest_delivered", 90
        else:
            organic_probability = float(
                np.clip(0.08 + 0.15 * event.affinity, 0.04, 0.30)
            )
            outcome = (
                "ORGANIC_INBOX_OPEN"
                if draw < organic_probability
                else "NO_OBSERVATION"
            )
            reward = 0.16 if outcome == "ORGANIC_INBOX_OPEN" else 0.0
            channel, delay = "no_notification_sent", 240
        return {
            "action_taken": ACTIONS[action],
            "channel": channel,
            "outcome": outcome,
            "delay_minutes": delay,
            "reward": round(float(reward), 4),
        }

    @staticmethod
    def teacher_distribution(
        event: Event,
        action: int,
        feedback: dict,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Return privileged teacher π(.|x,z), never oracle y*.

        The factual reward tilts only the route that was actually executed.
        """
        z = event.z
        urgency = event.importance * event.deadline
        scores = np.array(
            [
                1.30 * urgency
                + 0.38 * event.affinity
                - 1.05 * z["busy"]
                + 0.88 * z["incident_on_call"]
                + 0.48 * z["manager_focus"]
                + 0.42 * z["leisure_social"],
                0.66 * event.importance
                + 0.52 * event.affinity
                - 0.55 * urgency
                + 0.18 * z["busy"]
                - 0.48 * z["incident_on_call"],
                0.65 * (1 - event.importance)
                + 0.31 * (1 - event.affinity)
                - 0.68 * urgency
                - 0.40 * z["leisure_social"],
            ]
        )
        scores += rng.normal(0, 0.13, len(ACTIONS))
        scores[action] += 0.48 * float(feedback["reward"])
        return softmax(scores, temperature=TEACHER_TEMPERATURE)


DEFAULT_ENVIRONMENT = NotificationRoutingEnvironment()


# Compatibility functions keep the public API small while tests and notebooks
# can dependency-inject NotificationRoutingEnvironment directly.
def category_profile(category: str) -> tuple[float, float, float]:
    return DEFAULT_ENVIRONMENT.category_profile(category)


def make_event(
    rng: np.random.Generator,
    phase: int,
    index: int,
    prefix: str,
) -> Event:
    return DEFAULT_ENVIRONMENT.make_event(rng, phase, index, prefix)


def make_stream(seed: int) -> list[Event]:
    return DEFAULT_ENVIRONMENT.make_stream(seed)


def student_observation(event: Event) -> StudentObservation:
    return DEFAULT_ENVIRONMENT.student_observation(event)


def context_text(event: Event) -> str:
    return student_observation(event).text


def oracle_utilities(event: Event) -> np.ndarray:
    return DEFAULT_ENVIRONMENT.oracle_utilities(event)


def factual_feedback(
    event: Event,
    action: int,
    rng: np.random.Generator,
) -> dict:
    return DEFAULT_ENVIRONMENT.execute(event, action, rng)


def teacher_policy(
    event: Event,
    action: int,
    feedback: dict,
    rng: np.random.Generator,
) -> np.ndarray:
    return DEFAULT_ENVIRONMENT.teacher_distribution(event, action, feedback, rng)
