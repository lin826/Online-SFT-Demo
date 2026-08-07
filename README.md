# Online SDFT for Notification Routing

Picture one notification arriving **now**. The system knows its sender, urgency, timing, and the user's history, but it cannot know how the user will react until it chooses what to do. It must route the notification as `INTERRUPT`, `LATER`, or `ARCHIVE`; that choice changes what can be observed next.

That is a contextual bandit, not ordinary labeled classification. The agent repeatedly **predicts, acts, observes one factual outcome, and learns for the future**. It never gets to inspect all three possible futures.

![Contextual-bandit interaction: predict, act, observe, then learn for the next decision](figures/contextual_bandit_loop.png)

## The contextual-bandit contract

At interaction $t$:

1. A context $x_t$ arrives with information available **before** routing.
2. The student samples an action $a_t \sim \pi_t(\cdot \mid x_t)$, without current feedback or privileged information.
3. The action is recorded and scored immediately. This freezes the online result before learning can occur.
4. The environment executes **only** $a_t$, then returns factual feedback $z_t \sim P(\cdot \mid x_t,a_t)$.
5. A post-decision teacher reads $(x_t,a_t,z_t)$ and returns a soft policy $q_t=\pi_{\mathrm{teacher}}(\cdot \mid x_t,a_t,z_t)$.
6. A tiny online update may change $\pi_{t+1}$. It cannot rewrite the prediction already made at $t$.

The information boundary is the point of the experiment:

| Moment | Learner may use | Learner may **not** use |
| --- | --- | --- |
| Choose $a_t$ | $x_t$, current parameters, past factual records | current $z_t$, future events, oracle action, ground-truth demonstration |
| Observe | outcome caused by the selected $a_t$ | outcomes of either unchosen action |
| Update for $t+1$ | teacher distribution from $(x_t,a_t,z_t)$, small replay batch | simulator oracle, counterfactual user reactions, retroactive correction |
| Evaluate | a sealed scoring oracle computes correctness and regret | oracle output entering any policy or update function |

The scoring oracle is an **experiment instrument**, not a training label. It tells the researcher how costly the committed action was; it is never exposed to Base, ICL, RAG, Online-SFT, Online-SDFT, or the teacher.

> **Mental model:** every event is first a test decision and only afterward can become evidence for later decisions. There is no answer key before acting and no do-over after learning.

## Why the feedback is realistic

Feedback depends on the action actually taken. If the agent chooses `ARCHIVE`, no push or digest exists, so the only possible observations are an organic inbox open or no observation. A push click after `ARCHIVE` would be a fabricated counterfactual, and this simulator never generates one. Likewise, `NO_OBSERVATION` is ambiguous—not secretly converted into a negative ground-truth label.

The stream also drifts from weekday to on-call to off-hours behavior. The agent gets one pass through it, pays for cold start and exploration, and must adapt while serving decisions.

## How this differs from batch learning

| Batch supervised learning | This online contextual bandit |
| --- | --- |
| A fixed dataset contains $(x,y^*)$ labels before training | The stream reveals $x_t$, then feedback only after an action |
| Training can shuffle examples for many epochs | Events arrive once, in order, while user behavior drifts |
| The model is evaluated after training on a held-out split | Every live action is scored before its update |
| A label identifies the desired prediction | Feedback is partial and depends on the chosen action |
| Early training mistakes do not count toward test accuracy | Early mistakes remain in online accuracy and cumulative regret |

There is intentionally no train/test split: the objective is quality **during learning**, measured by prequential online accuracy and cumulative regret over the one stream.

## How regret is calculated

Online accuracy asks whether the chosen route exactly matches the simulator's best route. Regret is more informative: it measures **how much expected user utility the committed action leaves on the table**. Choosing a nearly equivalent route incurs little regret; choosing a harmful route when a much better one exists incurs more.

For evaluation, the simulator has an expected-utility function $\mu_t(a)$ for each action $a \in \mathcal A$. It uses the generated notification attributes and latent user state for round $t$. The scoring-only oracle is

$$
a_t^*=\underset{a \in \mathcal A}{\operatorname{arg\,max}}\;\mu_t(a).
$$

Once the policy commits to $a_t$, its instantaneous regret is

$$
r_t=\mu_t(a_t^*)-\mu_t(a_t) \ge 0.
$$

The experiment accumulates every decision—including cold start and 6% exploration—without resetting after an update:

$$
R_T=\sum_{t=1}^{T} r_t.
$$

### Concrete example

For the first weekday event in seed 0, the Base policy chose `INTERRUPT`. The simulator's sealed expected utilities were:

| Route | Expected utility $\mu_1(a)$ | Role |
| --- | ---: | --- |
| `INTERRUPT` | -0.5499 | policy action |
| `LATER` | 0.8528 | scoring-only oracle action |
| `ARCHIVE` | 0.5100 | unchosen alternative |

Therefore,

$$
r_1=0.8528-(-0.5499)=1.4027.
$$

The environment then executed only `INTERRUPT` and factually observed `IGNORED_PUSH`. It did **not** reveal what the user would have done under `LATER` or `ARCHIVE`. The full utility vector above is used by the benchmark evaluator only; it is never passed to the acting policy, teacher, replay memory, or update function. The selected action's sampled feedback may help the teacher construct $q_t$ for future rounds, but it cannot change $r_t$.

This separation is why the benchmark is not cheating: a simulator may retain all potential expected utilities to grade an agent, just as a game environment retains hidden state, while exposing only action-dependent observations to the learner. Exact regret would not be directly observable in a live notification system. A production study would estimate it through randomized propensity logging, an off-policy estimator, or an online controlled experiment. The realism claim here concerns the **agent's interaction and feedback path**; the complete simulator utility function exists only to make controlled evaluation possible.

Each method receives a cumulative result $R_{240}$ on each of 20 paired streams. The table reports the across-seed mean

$$
\bar R_{240}=\frac{1}{20}\sum_{s=1}^{20}R_{240}^{(s)}
$$

with a 95% confidence interval $\bar R_{240} \pm 1.96\,s_R/\sqrt{20}$. Regret is measured in simulator utility units, not clicks or percentage points. Lower is better; zero would mean selecting the scoring-best route on every round. `outputs/bandit/learning_curves.csv` contains $r_t$ and $R_t$ after every action, while `regret_per_decision` in the seed summaries is $R_{240}/240$.

## Where SDFT fits

The student generates its own rollout from $x_t$ alone. Only after the action and factual feedback does the privileged teacher produce $q_t$. Online-SDFT trains on that complete soft distribution with a small batch; Online-SFT receives only one sampled hard rollout from the same teacher. Neither method trains directly on the simulator's ground-truth action $y^*$.

## Result: one stream, predict then learn

There is no train/test split. Every action is scored before its feedback arrives, then that feedback can be used only for later decisions. Across 20 streams, Online-SDFT reaches **74.77% ± 1.24% online accuracy**, compared with 61.79% for Online-SFT, 53.15% for RAG, 52.17% for Base, and 45.75% for ICL. Its mean cumulative regret is **18.65**, versus 40.17 for the strongest learning baseline.

![Aggregate comparison](figures/bandit_accuracy.png)

![Online learning curves](figures/bandit_learning_curves.png)

## Reproduce

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python run.py
```

Or open [online_sdft_bandit_demo.ipynb](online_sdft_bandit_demo.ipynb) for the visual walkthrough. The notebook can regenerate all results or inspect the checked-in run artifacts.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lin826/Online-SFT-Demo/blob/main/online_sdft_bandit_demo.ipynb)

## Deliverables

- [TECHNICAL_REPORT.md](TECHNICAL_REPORT.md): formal problem setting, methods, results, and limitations.
- `outputs/bandit/rollouts.jsonl`: every rollout from Base, ICL, RAG, Online-SFT, and Online-SDFT.
- `outputs/bandit/learning_curves.csv`: raw per-step accuracy, cumulative accuracy, regret, and cumulative regret.
- `outputs/bandit/per_seed_metrics.csv`: one summary row per method and seed.
- `outputs/bandit/summary.json`: aggregate means, standard deviations, and confidence intervals.
- `outputs/bandit/qualitative_examples.json`: later-stage cases where SDFT is correct and all comparison arms are not.
- `figures/bandit_*.png`: aggregate, learning-curve, and action-feedback visualizations.
- `figures/contextual_bandit_loop.png`: causal online-interaction illustration used in this README and the notebook.

## Main files

| File | Purpose |
| --- | --- |
| `bandit_experiment.py` | Authoritative fast, multi-seed experiment and artifact generator |
| `online_sdft_bandit_demo.ipynb` | Visual general-audience walkthrough and replication notebook |
| `tests/test_bandit_experiment.py` | Causal feedback, information-boundary, and experiment invariants |

The multi-seed simulator is the primary reported experiment.
