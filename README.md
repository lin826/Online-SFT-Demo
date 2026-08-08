# Online SDFT for Notification Routing

An online contextual-bandit demo for routing notifications as `INTERRUPT`,
`LATER`, or `ARCHIVE`. A real
[LiquidAI LFM2.5-230M](https://huggingface.co/LiquidAI/LFM2.5-230M) student acts
before feedback, learns from a simulated post-decision teacher, and is scored on
the same drifting stream—there is no train/test split.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lin826/Online-SFT-Demo/blob/main/online_sdft_bandit_demo.ipynb)

![Online-SDFT causal interaction](figures/online_sdft_process.gif)

## Headline result

Mean over 3 paired 240-event streams; `±` is a 95% confidence interval.

| Method | Online accuracy | Cumulative regret ↓ |
| --- | ---: | ---: |
| Base | 37.08% ± 3.30 | 81.50 ± 2.24 |
| ICL | 37.08% ± 1.70 | 81.65 ± 0.87 |
| RAG | 38.61% ± 0.98 | 81.63 ± 6.75 |
| Online-SFT | 39.17% ± 5.10 | 102.82 ± 10.98 |
| **Online-SDFT** | **62.50% ± 5.66** | **43.33 ± 4.81** |

These preliminary results come from the real LFM student and replace the
earlier linear-policy proxy.

## Documentation

| Guide | Covers |
| --- | --- |
| [Problem setting](docs/problem-setting.md) | Causal feedback, information boundaries, and online versus batch learning |
| [Methods](docs/methods.md) | Base, ICL, RAG, Online-SFT, and Online-SDFT |
| [Evaluation and regret](docs/evaluation.md) | Exact regret calculation, utility weights, and their limitations |
| [Results and reproduction](docs/results.md) | Protocol, plots, commands, notebook, and artifacts |
| [Blog draft](BLOG.md) | An accessible narrative about continual, on-device learning and Online-SDFT |

## Repository map

| Path | Purpose |
| --- | --- |
| [`run.py`](run.py) | The supported experiment entry point |
| [`online_sdft/environment.py`](online_sdft/environment.py) | Stream simulation, factual feedback, rewards, and teacher |
| [`online_sdft/methods.py`](online_sdft/methods.py) | Liquid LFM policy and the five compared methods |
| [`online_sdft/experiment.py`](online_sdft/experiment.py) | Predict → score → execute → learn orchestration |
| [`online_sdft/reporting.py`](online_sdft/reporting.py) | Metrics, qualitative examples, and figures |
| [`online_sdft_bandit_demo.ipynb`](online_sdft_bandit_demo.ipynb) | Self-contained walkthrough and playable game |
| [`build_standalone_notebook.py`](build_standalone_notebook.py) | Rebuilds the notebook and process GIF |
| [`tests/`](tests) | Causal, method-boundary, and result invariants |

## Quick start

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python run.py
```

The first run downloads the Liquid model checkpoint from Hugging Face. Use
`--device cpu`, `--device mps`, or `--device cuda` to select the runtime.

[View the self-contained notebook on GitHub](online_sdft_bandit_demo.ipynb).
