# Results and Reproduction

[← Main README](../README.md)

## Protocol

| Item | Value |
| --- | --- |
| Streams | 20 paired random seeds |
| Decisions per stream | 240, in arrival order |
| Regimes | Weekday → on-call → off-hours |
| Exploration | 6%, included in all metrics |
| Methods | Base, ICL, RAG, Online-SFT, Online-SDFT |
| Primary metrics | Prequential online accuracy and cumulative regret |

Each seed creates one event stream that is reused across all methods. Every policy starts fresh, and every action is scored before its feedback or update.

## Aggregate results

Mean `±` 95% confidence interval:

| Method | Online accuracy | Cumulative regret ↓ |
| --- | ---: | ---: |
| Base | 52.17% ± 1.23 | 71.17 ± 2.93 |
| ICL | 45.75% ± 1.05 | 78.38 ± 3.39 |
| RAG | 53.15% ± 1.62 | 56.60 ± 4.39 |
| Online-SFT | 61.79% ± 2.51 | 40.17 ± 4.51 |
| **Online-SDFT** | **74.77% ± 1.24** | **18.65 ± 1.28** |

![Aggregate comparison](../figures/bandit_accuracy.png)

![Online learning curves](../figures/bandit_learning_curves.png)

Online-SDFT improves both objectives on the same streams. The result concerns performance **during adaptation**, not held-out accuracy after training.

## Reproduce

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python run.py
```

The [self-contained notebook](../online_sdft_bandit_demo.ipynb) embeds the simulator, methods, experiment runner, animation, plots, and audit. It reads no repository artifacts at runtime.

[Open in Colab](https://colab.research.google.com/github/lin826/Online-SFT-Demo/blob/main/online_sdft_bandit_demo.ipynb)

## Artifacts

| Path | Contents |
| --- | --- |
| `outputs/bandit/per_seed_metrics.csv` | One final row per method and seed |
| `outputs/bandit/summary.json` | Aggregate means and confidence intervals |
| `outputs/bandit/qualitative_examples.json` | Later-stream SDFT wins |
| `outputs/bandit/rollouts.jsonl` | Generated locally: every action, factual outcome, teacher distribution, and score |
| `outputs/bandit/learning_curves.csv` | Generated locally: per-step accuracy and regret |

Only the three compact result files are versioned. The two larger audit files are reproducible with `python run.py` and ignored by Git.
