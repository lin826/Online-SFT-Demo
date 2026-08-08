# Results and Reproduction

[← Main README](../README.md)

## Protocol

| Item | Value |
| --- | --- |
| Streams | 3 paired random seeds |
| Decisions per stream | 240, in arrival order |
| Regimes | Weekday → on-call → off-hours |
| Serving policy | LFM argmax with 6% uniform exploration |
| Student | LiquidAI/LFM2.5-230M |
| Trainable state | Rank-4 LoRA, 172,032 parameters |
| Published runtime | Apple MPS, one isolated model load per seed |
| Methods | Base, ICL, RAG, Online-SFT, Online-SDFT |
| Primary metrics | Prequential online accuracy and cumulative regret |

Each seed creates one event stream that is reused across all methods. Every policy starts fresh, and every action is scored before its feedback or update.

## Aggregate results

Mean `±` 95% confidence interval:

| Method | Online accuracy | Cumulative regret ↓ |
| --- | ---: | ---: |
| Base | 37.08% ± 3.30 | 81.50 ± 2.24 |
| ICL | 37.50% ± 1.25 | 81.10 ± 1.37 |
| RAG | 38.75% ± 0.47 | 79.94 ± 7.38 |
| Online-SFT | 41.25% ± 2.49 | 99.02 ± 13.52 |
| **Online-SDFT** | **63.75% ± 1.25** | **37.43 ± 0.96** |

![Aggregate comparison](../figures/bandit_accuracy.png)

![Online learning curves](../figures/bandit_learning_curves.png)

Online-SDFT improves both objectives on every paired stream. With only three
seeds, the confidence intervals are preliminary; the result concerns
performance **during adaptation**, not held-out accuracy after training.

## Reproduce

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python run.py
```

The [self-contained notebook](../online_sdft_bandit_demo.ipynb) embeds the
simulator, methods, experiment runner, animation, plots, and audit. It reads no
repository artifacts at runtime; its setup cell installs the runtime and
downloads the [Liquid model](https://huggingface.co/LiquidAI/LFM2.5-230M).

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
