# Online SDFT for Notification Routing

An online contextual-bandit demo for routing notifications as `INTERRUPT`,
`LATER`, or `ARCHIVE`. A real
[LiquidAI LFM2.5-230M](https://huggingface.co/LiquidAI/LFM2.5-230M) student acts
before feedback, learns from a simulated post-decision teacher, and is scored on
the same drifting stream—there is no train/test split.

**[Explore the interactive project website](https://lin826.github.io/Online-SFT-Demo/)**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lin826/Online-SFT-Demo/blob/main/online_sdft_bandit_demo.ipynb)

For a full rerun, select **Runtime → Change runtime type → T4 GPU** in Colab,
then execute Sections 6.1–6.4. The setup cell preserves Colab's CUDA PyTorch and
removes its incompatible, unused `torchao` package before loading PEFT. The
verified T4 workflow includes a pass/fail ranking check.

![Online-SDFT causal interaction](figures/online_sdft_process.gif)

## Headline result

Mean over 3 paired 240-event streams; `±` is a 95% confidence interval.

| Method | Online accuracy | Cumulative regret ↓ |
| --- | ---: | ---: |
| Base | 37.08% ± 3.30 | 81.50 ± 2.24 |
| ICL | 37.50% ± 1.25 | 81.10 ± 1.37 |
| RAG | 38.75% ± 0.47 | 79.94 ± 7.38 |
| REINFORCE | 32.08% ± 1.70 | 115.65 ± 16.88 |
| Online-SFT | 41.94% ± 2.72 | 97.65 ± 13.23 |
| **Online-SDFT** | **64.72% ± 3.14** | **36.24 ± 1.66** |

These preliminary results come from the real LFM student and replace the
earlier linear-policy proxy.

## Documentation

| Guide | Covers |
| --- | --- |
| [Problem setting](docs/problem-setting.md) | Causal feedback, information boundaries, and online versus batch learning |
| [Methods](docs/methods.md) | Algorithms plus complete ICL/RAG prompts and actual LFM responses |
| [Evaluation and regret](docs/evaluation.md) | Exact regret calculation, utility weights, and their limitations |
| [Results and reproduction](docs/results.md) | Protocol, plots, commands, notebook, and artifacts |
| [Blog draft](BLOG.md) | An accessible narrative about continual, on-device learning and Online-SDFT |

## Repository map

| Path | Purpose |
| --- | --- |
| [`run.py`](run.py) | The supported experiment entry point |
| [`online_sdft/environment.py`](online_sdft/environment.py) | Stream simulation, factual feedback, rewards, and teacher |
| [`online_sdft/methods.py`](online_sdft/methods.py) | Liquid LFM policy and the six compared methods |
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
