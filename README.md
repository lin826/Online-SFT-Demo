# Online SDFT for Notification Routing

A router chooses `INTERRUPT`, `LATER`, or `ARCHIVE` before seeing the user's reaction. Every action counts; feedback can improve only future decisions.

![Animated Online-SDFT causal round](figures/online_sdft_process.gif)

## Contents

- [Setting](#setting)
- [Methods](#methods)
- [Regret and utility choices](#regret-and-utility-choices)
- [Results](#results)
- [Run it](#run-it)

## Setting

```text
context → commit action → freeze score → execute action → observe feedback → update t+1
```

`ARCHIVE` sends nothing, so it returns only `ORGANIC_INBOX_OPEN` or `NO_OBSERVATION`—never a fictional click.

The learner never receives current feedback, unchosen outcomes, oracle utilities, or ground-truth demonstrations. With no train/test split, every mistake stays in the drifting 240-event score.

## Methods

| Method | Online adaptation |
| --- | --- |
| Base | Frozen policy |
| ICL | Recent hard teacher rollouts in context |
| RAG | Similar hard rollouts retrieved |
| Online-SFT | Updates from one sampled teacher action |
| **Online-SDFT** | Acts without `z_t`; updates ≤4 records from the teacher's soft distribution, not oracle labels |

## Regret and utility choices

**Accuracy is binary; regret is not.** Accuracy checks whether the route equals the evaluator's best route. Regret measures the expected utility lost:

```text
step_regret(t) = max_a utility(t, a) - utility(t, chosen_action)
cumulative_regret(T) = sum_t step_regret(t)
```

Only the chosen action's factual outcome reaches the teacher. The three-action utility vector stays inside the evaluator. Regret therefore captures urgency, interruption cost, delay, and suppression—not only click/no-click.

Let `I` = importance, `D` = deadline pressure, `F` = affinity, `B` = busyness, and `U = I×D`; `C/M/S` flag on-call incident, manager-focus, and off-hours social contexts.

| Route | Utility | Why these terms have this direction and scale |
| --- | --- | --- |
| `INTERRUPT` | `1.45U + 0.42F − 1.20B + 1.00C + 0.60M + 0.50S` | Urgency is the main benefit; busyness is the main cost. Context boosts cover cases where delay is unusually harmful. |
| `LATER` | `0.72I + 0.58F − 0.62U + 0.22B − 0.62C` | Relevance preserves the item; urgency and incidents penalize delay; busyness mildly favors deferral. |
| `ARCHIVE` | `0.72(1−I) + 0.36(1−F) − 0.80U − 0.50S` | Low-value items are safe to suppress; urgent or desired social items are costly to hide. |

The selected action has a separate teacher reward: interrupt open/dismiss/ignore = `+.72/−.58/−.78`, then `−.30B`; later open/ignore = `+.48/−.16`, then `−.28U`; archive organic-open/otherwise = `+.16/0`. This makes interruption high-upside/high-cost, delay urgency-sensitive, and silence neutral. It never enters regret.

These are **simulation assumptions, not production estimates**: urgency may justify interruption, busyness may outweigh relevance, and archiving requires low value. The oracle mix (`INTERRUPT/LATER/ARCHIVE`) is 34%/38%/28%. Utility scale is arbitrary; production weights require experiments and stakeholder costs.

Exact implementation: [`oracle_utilities`](bandit_experiment.py#L141-L152) and [`run_method`](bandit_experiment.py#L257-L335).

```text
a_t = sample(student(x_t, past))
u = oracle_utilities(hidden_state)      # evaluator only
regret += max(u) - u[a_t]               # frozen before feedback
z_t = execute_only(a_t)
update_t_plus_1(teacher(x_t, a_t, z_t))
```

## Results

Mean over 20 paired streams; `±` is a 95% confidence interval.

| Method | Online accuracy | Cumulative regret ↓ |
| --- | ---: | ---: |
| Base | 52.17% ± 1.23 | 71.17 ± 2.93 |
| ICL | 45.75% ± 1.05 | 78.38 ± 3.39 |
| RAG | 53.15% ± 1.62 | 56.60 ± 4.39 |
| Online-SFT | 61.79% ± 2.51 | 40.17 ± 4.51 |
| **Online-SDFT** | **74.77% ± 1.24** | **18.65 ± 1.28** |

![Aggregate comparison](figures/bandit_accuracy.png)

## Run it

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python run.py
```

The [self-contained notebook](online_sdft_bandit_demo.ipynb) embeds the full experiment.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lin826/Online-SFT-Demo/blob/main/online_sdft_bandit_demo.ipynb)

Details: [technical report](TECHNICAL_REPORT.md), `outputs/bandit/`, and [causal tests](tests/test_bandit_experiment.py).
