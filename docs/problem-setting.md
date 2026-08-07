# Problem Setting

[← Main README](../README.md)

## A contextual bandit, not labeled classification

At round `t`, a context `x_t` arrives and the student samples a route from its current policy. The route is committed and scored before the environment reveals anything about the user's response.

```text
context → commit action → freeze score → execute action → observe feedback → update t+1
```

| Moment | Available to the learner | Sealed away |
| --- | --- | --- |
| Choose `a_t` | `x_t`, current parameters, past factual records | Current feedback, future events, scoring oracle |
| Observe | Outcome caused by `a_t` | Outcomes of the two unchosen actions |
| Update `t+1` | Post-decision teacher target, batch of ≤4 records | Oracle action, ground-truth demonstration, retroactive score changes |

The evaluator may inspect simulator state to grade the frozen action, but neither the student nor teacher receives its utility vector or preferred route.

## Action-dependent feedback

The environment executes only the selected route:

| Action | What exists | Possible factual outcomes |
| --- | --- | --- |
| `INTERRUPT` | Immediate push | `OPENED_PUSH`, `DISMISSED_PUSH`, `IGNORED_PUSH` |
| `LATER` | Digest after 90 minutes | `OPENED_DIGEST`, `IGNORED_DIGEST` |
| `ARCHIVE` | No notification | `ORGANIC_INBOX_OPEN`, `NO_OBSERVATION` |

Therefore, `ARCHIVE` can never produce a push or digest click. `NO_OBSERVATION` is ambiguous rather than being converted into a hidden negative label. This avoids pretending that a historical response under one action reveals what would have happened under another.

## Why this is online learning

| Batch supervised learning | This experiment |
| --- | --- |
| Labels exist before training | Feedback exists only after acting |
| Examples can be shuffled for many epochs | Events arrive once, in order |
| Evaluation follows training | Every action is evaluated during learning |
| Early training errors do not enter test accuracy | Cold-start, exploration, and adaptation errors all count |

There is intentionally no train/test split. Each 240-event stream moves through weekday, on-call, and off-hours regimes. The goal is to maximize **prequential online accuracy** and minimize **cumulative regret** while serving that one drifting stream.

See [Methods](methods.md) for the learners and [Evaluation and regret](evaluation.md) for the sealed scoring oracle.
