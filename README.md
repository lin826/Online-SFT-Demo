# Online SDFT for Notification Routing

This repository demonstrates Online Soft-Distillation Fine-Tuning in a realistic notification-routing contextual bandit. A policy chooses `INTERRUPT`, `LATER`, or `ARCHIVE`; only that action is executed, and only its factual outcome is observed. Archived notifications never produce fictional clicks.

The privileged teacher sees post-decision feedback (z) and emits a soft rollout distribution `π_teacher(.|x,z)`. The on-device student acts from `x` alone and learns by KL divergence. It never trains on the simulator's evaluation-only oracle action.

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

## Main files

| File | Purpose |
| --- | --- |
| `bandit_experiment.py` | Authoritative fast, multi-seed experiment and artifact generator |
| `online_sdft_bandit_demo.ipynb` | Visual general-audience walkthrough and replication notebook |
| `tests/test_bandit_experiment.py` | Causal feedback, information-boundary, and experiment invariants |

The multi-seed simulator is the primary reported experiment.
