# Online SDFT for Censored Notification Feedback

## Executive summary

This experiment evaluates notification routing as a pure online-learning problem. There is one stream and no train/test split. At every step, each agent must choose `INTERRUPT`, `LATER`, or `ARCHIVE`; that action is scored immediately, then executed, then its factual feedback becomes available for future learning.

Online Soft-Distillation Fine-Tuning (Online-SDFT) learns from a privileged teacher distribution `π_teacher(.|x,z)`. The online student acts from `x` alone and never learns from the simulator's scoring-only oracle action.

Across 20 paired streams, Online-SDFT achieves the highest prequential online accuracy and the lowest cumulative contextual-bandit regret:

| Method | Online accuracy ↑ | Cumulative regret ↓ |
| --- | ---: | ---: |
| Base | 52.17% ± 1.23 | 71.17 ± 2.93 |
| ICL | 45.75% ± 1.05 | 78.38 ± 3.39 |
| RAG | 53.15% ± 1.62 | 56.60 ± 4.39 |
| Online-SFT | 61.79% ± 2.51 | 40.17 ± 4.51 |
| **Online-SDFT** | **74.77% ± 1.24** | **18.65 ± 1.28** |

Values are means ±95% confidence intervals over seeds. Relative to Online-SFT, Online-SDFT gains 12.98 ±2.68 accuracy points and reduces regret by 21.52 ±4.92. It wins online accuracy in all 20 paired seeds and regret in 19 of 20.

![Online accuracy and regret](figures/bandit_accuracy.png)

## 1. One-stream online problem

At time (t):

1. A notification context (x_t) arrives.
2. The agent chooses (a_t\sim\pi_t(\cdot\mid x_t)).
3. Before seeing feedback, its action is scored for online accuracy and regret.
4. Only (a_t) is executed.
5. Factual feedback (z_t\sim P(z\mid x_t,a_t)) arrives.
6. The method may update and use that information starting at (t+1).

Thus the reported accuracy is prequential:

\[
\mathrm{OnlineAccuracy}_T=\frac1T\sum_{t=1}^T
\mathbf1[a_t=a_t^*].
\]

There is no held-out evaluation and no retroactive rescoring of an updated model.

The action space is

\[
\mathcal A=\{\texttt{INTERRUPT},\texttt{LATER},\texttt{ARCHIVE}\}.
\]

The 240-decision stream drifts through three consecutive 80-decision regimes: weekday, on-call, and off-hours. Each context contains a notification category, importance, deadline pressure, affinity, time-of-day, coarse time-of-week, and feature interactions. Seven categories are balanced: manager, calendar, monitoring, teammate, social, receipt, and promotion.

## 2. Realistic censored feedback

Only the selected route can produce an outcome:

| Selected route | Observable factual outcomes |
| --- | --- |
| `INTERRUPT` | `OPENED_PUSH`, `DISMISSED_PUSH`, `IGNORED_PUSH` |
| `LATER` | `OPENED_DIGEST`, `IGNORED_DIGEST` after a delay |
| `ARCHIVE` | `ORGANIC_INBOX_OPEN`, `NO_OBSERVATION` |

An archived notification never produces a push or digest click because no notification was sent. `NO_OBSERVATION` is ambiguous rather than a negative ground-truth label. The simulator does not sample or expose outcomes for either unchosen action.

The simulator contains expected utilities only to score decisions. The scoring-only oracle is

\[
a_t^*=\arg\max_a\mu(x_t,a),
\]

and per-step regret is

\[
\Delta_t=\mu(x_t,a_t^*)-\mu(x_t,a_t).
\]

Cumulative regret is (R_T=\sum_{t=1}^T\Delta_t). Oracle fields in the raw logs are explicitly suffixed `scoring_only`; they are never passed into any update function.

## 3. Teacher rollouts—not ground truth

After the action and factual outcome, a privileged teacher observes ((x_t,a_t,z_t)). Its extra telemetry includes current device interruptibility and semantic metadata that are realistic for a more capable post-hoc/cloud teacher but unavailable to the lightweight pre-decision student.

The teacher emits a noisy calibrated distribution

\[
q_t(a)=\pi_{\mathrm{teacher}}(a\mid x_t,a_t,z_t),
\]

and a hard teacher rollout is sampled as (	ilde a_t\sim q_t). The teacher is not handed (a_t^*).

All agents use an epsilon-greedy rollout with 6% exploration. Explored actions remain in both accuracy and regret; they are not filtered from results.

## 4. Compared methods

Every method starts from the same weak generic linear softmax policy.

### Base

The policy remains frozen. It receives no history.

### ICL

The policy blends its base distribution with hard teacher rollouts from its 12 most recent factual interactions. This is a recency context with no weight update.

### RAG

The policy retrieves the five closest earlier contexts and blends their hard teacher rollout votes with the base distribution. Retrieval can use stale actions after a regime shift.

### Online-SFT

SFT uses one sampled hard teacher rollout as a one-hot target:

\[
L_{\mathrm{SFT}}=-\log\pi_\theta(\tilde a_t\mid x_t).
\]

This is not direct training on (a_t^*). It is hard distillation from one noisy teacher sample.

### Online-SDFT

SDFT preserves all probabilities:

\[
L_{\mathrm{SDFT}}=
D_{\mathrm{KL}}\left(q_t(\cdot)\;\|\;\pi_\theta(\cdot\mid x_t)\right).
\]

SFT and SDFT both use a 24-record sliding replay window and a batch of at most four records: the fresh item plus three replay samples. Separate coarse learning-rate sweeps select stable rates for hard and soft targets; the soft distribution permits a larger update without the variance of sampled one-hot labels.

This is the key controlled comparison: SFT observes one draw from (q_t); SDFT observes the full (q_t).

## 5. Online results

![Online learning curves](figures/bandit_learning_curves.png)

The left panel plots cumulative online accuracy after every prediction; the right plots cumulative regret. Both include early cold-start and exploration costs.

Accuracy by the three consecutive stream segments is:

| Method | Weekday | On-call | Off-hours |
| --- | ---: | ---: | ---: |
| Base | 40.56% | 55.06% | 60.88% |
| ICL | 45.88% | 44.94% | 46.44% |
| RAG | 45.13% | 56.44% | 57.88% |
| Online-SFT | 57.50% | 62.12% | 65.75% |
| **Online-SDFT** | **68.38%** | **77.94%** | **78.00%** |

SDFT leads even in the first 80-decision segment and expands the advantage as interactions accumulate. Its full teacher distribution supplies relative evidence about all three actions on every update, whereas SFT discards that information when it samples one hard rollout.

## 6. Qualitative online decisions

The exported examples occur only after the first 80 decisions. They select positions where SDFT's actual pre-feedback action matches the scoring oracle and every comparison arm's actual action does not.

- **Step 87, on-call receipt:** Base and RAG interrupt; ICL and SFT defer; SDFT archives. Because it archives, its only factual result is `NO_OBSERVATION`—not a fictional click.
- **Step 150, on-call receipt:** SDFT again archives after accumulating soft evidence across similar events; hard-history approaches continue to interrupt or defer.
- **Step 162, off-hours social:** SDFT chooses `LATER`; all other arms interrupt. Their own factual feedback differs because each action is executed independently.

The exact action, delivery channel, outcome, delay, reward, and teacher rollout for every method are in [`outputs/bandit/qualitative_examples.json`](outputs/bandit/qualitative_examples.json). Selection happens after rollouts are complete and cannot affect training.

## 7. Raw artifacts and reproduction

Run:

```bash
.venv/bin/python bandit_experiment.py --seeds 20
```

| Artifact | Evidence |
| --- | --- |
| `outputs/bandit/rollouts.jsonl` | 24,000 factual decisions: 20 seeds × 5 methods × 240 steps |
| `outputs/bandit/learning_curves.csv` | per-step correctness, regret, cumulative accuracy, cumulative regret |
| `outputs/bandit/per_seed_metrics.csv` | one final online result per method and seed |
| `outputs/bandit/summary.json` | mean, standard deviation, and 95% CI |
| `outputs/bandit/qualitative_examples.json` | later-stream examples unique to SDFT |
| `figures/bandit_accuracy.png` | headline online metrics |
| `figures/bandit_learning_curves.png` | raw online learning trajectories |
| `figures/bandit_action_feedback.png` | actions actually executed |
| `online_sdft_bandit_demo.ipynb` | visual replication notebook |

## 8. Limitations

This is a controlled simulator, not deployed-user evidence. The privileged teacher is engineered to have noisy but useful post-decision telemetry. The result supports a specific mechanism: when a calibrated teacher distribution is available after factual interaction, preserving the entire distribution can be much more sample-efficient than learning from one sampled rollout.

A production evaluation should use randomized propensity logging and the same prequential metrics. It should never construct labels from unobserved counterfactual user behavior.
