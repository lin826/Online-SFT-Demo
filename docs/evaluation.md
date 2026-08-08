# Evaluation and Regret

[← Main README](../README.md)

## Three different signals

| Signal | Form | Used by |
| --- | --- | --- |
| Online accuracy | Binary match with the evaluator's best route | Evaluation only |
| Factual feedback | One sampled outcome and reward for the chosen route | Post-decision teacher |
| Regret | Real-valued utility gap | Evaluation only |

Accuracy says whether the selected action was best. Regret says **how costly the mistake was**:

```text
step_regret(t) = max_a utility(t, a) - utility(t, chosen_action)
cumulative_regret(T) = sum_t step_regret(t)
```

A near-tie incurs little regret; interrupting a busy user when deferral is much better incurs more. The full utility vector is never supplied to the student, teacher, memory, or gradient update.

## Evaluator utility

Let `I` = importance, `D` = deadline pressure, `F` = affinity, `B` = latent busyness, and `U = I×D`. `C`, `M`, and `S` indicate an on-call incident, manager-focus event, and off-hours social event.

| Route | Expected utility | Design rationale |
| --- | --- | --- |
| `INTERRUPT` | `1.45U + 0.42F − 1.20B + 1.00C + 0.60M + 0.50S` | Urgency is the main benefit; busyness is the main cost. Context boosts cover cases where delay is unusually harmful. |
| `LATER` | `0.72I + 0.58F − 0.62U + 0.22B − 0.62C` | Relevance preserves the item; urgency and incidents penalize delay; busyness mildly favors deferral. |
| `ARCHIVE` | `0.72(1−I) + 0.36(1−F) − 0.80U − 0.50S` | Low-value items are safe to suppress; urgent or desired social items are costly to hide. |

These coefficients encode three product assumptions: urgency can justify interruption, busyness can outweigh ordinary relevance, and archiving should require low value. Their relative scale lets a strong interruption cost offset a routine relevance benefit.

They are **transparent simulation assumptions, not production estimates**. The
resulting scoring-best action mix across the three canonical streams is
nontrivial: 33.2% `INTERRUPT`, 37.5% `LATER`, and 29.3% `ARCHIVE`.
Absolute utility units are arbitrary—multiplying all coefficients by a constant
would rescale regret without changing the preferred action. Comparisons are
therefore meaningful only within this fixed benchmark.

## Factual reward used by the teacher

The chosen route also returns a smaller reward that tilts only its teacher score:

| Action | Outcome reward | Additional cost | Rationale |
| --- | --- | --- | --- |
| Interrupt | open `+0.72`; dismiss `−0.58`; ignore `−0.78` | `−0.30B` | Highest upside and highest disruption risk |
| Later | open `+0.48`; ignore `−0.16` | `−0.28U` | Gentler delivery, but delay hurts urgent items |
| Archive | organic open `+0.16`; otherwise `0` | none | Silence is neutral; organic discovery is weak evidence |

This factual reward affects future teacher supervision; it does **not** enter regret. Keeping the two signals separate prevents observed engagement from becoming a counterfactual oracle label.

## Exact ordering in code

```text
a_t = epsilon_greedy(student(x_t, past), epsilon=0.06)
u_t = oracle_utilities(hidden_simulator_state)   # evaluator only
best_t = argmax(u_t)
step_regret = u_t[best_t] - u_t[a_t]             # score is frozen

z_t = execute_only(a_t)                          # one factual outcome
q_t = teacher(x_t, a_t, z_t)
update_for_t_plus_1(q_t)
```

The simulator-only utilities and factual execution live in
[`environment.py`](../online_sdft/environment.py), the chronological ordering
is enforced in [`experiment.py`](../online_sdft/experiment.py), and LFM/LoRA
updates live in [`methods.py`](../online_sdft/methods.py).

For each method, the reported cumulative regret is the mean final `R_240`
across three paired streams; confidence intervals use
`1.96 × sample_std / sqrt(3)`. Every cold-start and exploration action remains
included.

## Production interpretation

Exact counterfactual regret is available here because this is a simulator. A deployed system would need randomized propensity logging, off-policy estimators, or controlled experiments. Product teams would also need to calibrate interruption, delay, and suppression costs with users and stakeholders rather than reusing these demonstration weights.
