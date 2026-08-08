# Cursor Handoff: Online-SDFT Notification Routing

Last updated: 2026-08-07  
Repository: <https://github.com/lin826/Online-SFT-Demo>  
Live article: <https://lin826.github.io/Online-SFT-Demo/>  
Colab: <https://colab.research.google.com/github/lin826/Online-SFT-Demo/blob/main/online_sdft_bandit_demo.ipynb>

## Current state

- Branch: `main`
- Latest implementation commit before this handoff: `b0511c6` (`Make article copy more professional`)
- Git remote: `origin` → `git@github-lin826:lin826/Online-SFT-Demo.git`
- The working tree was clean before `HANDOFF.md` was added.
- GitHub Pages is enabled with **GitHub Actions** as the source.
- Latest verified Pages deployment: <https://github.com/lin826/Online-SFT-Demo/actions/runs/31236905071>
- The last full local check passed: **28 tests**.

The project is functional and published. There is no known blocking bug.

## User's non-negotiable requirements

1. This is an **online/streaming learning** experiment, not batch training.
   There is one chronological stream and no train/test split. Optimize and
   report prequential **online accuracy** and **cumulative regret**.
2. The notification environment must be causal and realistic. The model
   selects one route and receives only the factual feedback that route can
   produce. It must never receive outcomes for unselected routes.
3. If the action is `ARCHIVE`, there was no push or digest. Its only valid
   outcomes are `ORGANIC_INBOX_OPEN` and `NO_OBSERVATION`.
4. Online-SDFT must satisfy all three boundaries:
   - learn from a teacher distribution `pi(. | x, z)`, not directly from an
     evaluator's ground-truth action;
   - update online in small batches;
   - generate its own rollout from visible `x`, without privileged `z`.
5. Use a real language model: `LiquidAI/LFM2.5-230M`. Do not replace it with a
   linear proxy.
6. Baselines are Base, ICL, RAG, REINFORCE, Online-SFT, and Online-SDFT. ICL
   and RAG should be implemented in good faith and explained with real prompt
   traces.
7. The notebook must be self-contained and reproducible on a Colab T4 GPU.
8. Keep the repository focused on this notification-routing experiment. Do not
   reintroduce obsolete scripts, figures, or alternate demos.
9. Push completed changes to GitHub and verify the Pages deployment.
10. Editorial tone must be professional. The user explicitly rejected copy
    such as “the benchmark measures the journey, not a held-out exam.” Avoid
    cute metaphors, jokes, anthropomorphism, or inflated claims. Prefer terms
    such as “prequential evaluation,” “action-dependent feedback,” and
    “information boundary.”

## Problem contract

At round `t`:

```text
observe x_t
→ student chooses a_t without z_t
→ evaluator freezes the pre-update score (scoring only)
→ environment executes only a_t
→ factual feedback for a_t is observed
→ post-decision teacher returns q_t = pi(. | x_t, a_t, z_t, feedback_t)
→ method updates for round t+1
```

Actions:

| Code | Route | Possible factual outcomes |
| --- | --- | --- |
| A | `INTERRUPT` | `OPENED_PUSH`, `DISMISSED_PUSH`, `IGNORED_PUSH` |
| B | `LATER` | `OPENED_DIGEST`, `IGNORED_DIGEST` |
| C | `ARCHIVE` | `ORGANIC_INBOX_OPEN`, `NO_OBSERVATION` |

Important separation:

- `Event` contains simulator-only `z` and must not cross into methods.
- `StudentObservation` is the complete student view.
- `oracle_utilities(event)` exists only to calculate online accuracy and
  regret. Its vector and best action must never enter a prompt, replay record,
  teacher target, reward, or gradient.
- `teacher_distribution(...)` is a stochastic post-decision teacher and is
  intentionally not the evaluator oracle.

## Experiment configuration

Defined in `online_sdft/config.py`:

| Setting | Current value |
| --- | --- |
| Model | `LiquidAI/LFM2.5-230M` |
| Routes | `INTERRUPT`, `LATER`, `ARCHIVE` |
| Stream length | 240 decisions |
| Regimes | weekday → on-call → off-hours |
| Decisions per regime | 80 |
| Published seeds | 3 (`0, 1, 2`) |
| Non-RL serving | 6% epsilon-greedy |
| REINFORCE serving | sample from LFM policy |
| LoRA | rank 4, alpha 8, 172,032 trainable parameters |
| Online replay | 24 records |
| Online batch size | 4 |
| ICL examples | 12 recent records |
| RAG examples | 12 nearest records |
| SFT learning rate | `2e-4` |
| SDFT learning rate | `3e-4` |
| REINFORCE learning rate | `1e-4` |
| Teacher temperature | `0.95` |

The student policy is the softmax over the LFM's next-token logits for the
single-token codes `A`, `B`, and `C`.

## Method implementations

All method code is in `online_sdft/methods.py`.

- **Base**: frozen LFM, no memory.
- **ICL**: frozen LFM; prompt contains the last 12 legal post-decision teacher
  examples, ordered oldest to newest.
- **RAG**: frozen LFM; searches all legal past records using an equal-weight
  Gower-style similarity over category, regime, importance, deadline,
  affinity, and circular hour. The top 12 are ordered weaker-to-stronger so the
  best match is adjacent to the query. Retrieval never uses the label, hidden
  `z`, reward, or evaluator utility.
- **REINFORCE**: batch-one LoRA policy-gradient update from the selected
  action's scalar factual reward only. It uses a past-only EMA baseline
  (`step=0.05`) and entropy coefficient `0.01`; it never queries the teacher.
- **Online-SFT**: samples one teacher action and trains on its one-hot target.
- **Online-SDFT**: trains on the teacher's complete soft distribution.

ICL and RAG store their record only after the current action, execution, and
teacher response. The complete trace-backed prompts and actual LFM responses
for seed 0, decision 148 are in `docs/methods.md` and the notebook.

## Published results

Mean over three paired 240-event streams. `±` is a 95% confidence interval.

| Method | Online accuracy | Cumulative regret ↓ |
| --- | ---: | ---: |
| Base | 37.08% ± 3.30 | 81.50 ± 2.24 |
| ICL | 37.50% ± 1.25 | 81.10 ± 1.37 |
| RAG | 38.75% ± 0.47 | 79.94 ± 7.38 |
| REINFORCE | 32.08% ± 1.70 | 115.65 ± 16.88 |
| Online-SFT | 41.94% ± 2.72 | 97.65 ± 13.23 |
| **Online-SDFT** | **64.72% ± 3.14** | **36.24 ± 1.66** |

These are preliminary simulator results. Do not describe three seeds as a
definitive empirical conclusion.

Canonical compact artifacts:

- `outputs/bandit/per_seed_metrics.csv`
- `outputs/bandit/summary.json`
- `outputs/bandit/qualitative_examples.json`

Generated `rollouts.jsonl` and `learning_curves.csv` may exist locally but are
ignored by Git. Do not commit them unless the repository policy is deliberately
changed.

## Regret

The evaluator computes:

```text
step_regret(t) = max_a utility(t, a) - utility(t, chosen_action)
cumulative_regret(T) = sum_t step_regret(t)
```

Regret is not binary. It uses the three real-valued evaluator utilities defined
in `NotificationRoutingEnvironment.oracle_utilities`. The coefficients and
their rationale are documented in `docs/evaluation.md`. They are transparent
simulation assumptions, not production estimates. Absolute utility units are
arbitrary and comparisons are valid only within this fixed benchmark.

Factual route reward is a separate signal. It is used by REINFORCE and tilts
the executed route in the teacher; it does not enter regret.

## Repository map

| Path | Responsibility |
| --- | --- |
| `run.py` | Supported CLI entry point |
| `online_sdft/config.py` | Names and experiment hyperparameters |
| `online_sdft/environment.py` | Stream, `Event`, student projection, execution, rewards, evaluator utilities, teacher |
| `online_sdft/methods.py` | Liquid LFM policy, prompts, ICL/RAG, REINFORCE, SFT, SDFT |
| `online_sdft/experiment.py` | Causal predict → score → execute → teach → update loop |
| `online_sdft/reporting.py` | Aggregation, examples, figures |
| `bandit_experiment.py` | Compatibility facade; new code should import focused modules |
| `build_standalone_notebook.py` | Canonical notebook/GIF builder |
| `online_sdft_bandit_demo.ipynb` | Self-contained walkthrough and Colab experiment |
| `docs/problem-setting.md` | Contextual-bandit and information-boundary explanation |
| `docs/methods.md` | Exact algorithms and trace-backed prompts |
| `docs/evaluation.md` | Utility rationale and exact regret calculation |
| `docs/results.md` | Protocol, results, artifacts, reproduction |
| `BLOG.md` | Accessible long-form blog draft |
| `website/` | Static site generated from `BLOG.md` by `build_site.py` |
| `.github/workflows/pages.yml` | GitHub Pages assembly and deployment |
| `tests/` | Causal, method, result, and notebook invariants |

## Notebook and Colab

The notebook must remain self-contained: it must not read repository files at
runtime. `build_standalone_notebook.py` embeds the focused Python modules into
the notebook and generates `figures/online_sdft_process.gif`.

Colab-specific constraints already handled:

- requires a GPU in Colab and reports the selected device;
- preserves Colab's CUDA PyTorch;
- removes the incompatible, unused preinstalled `torchao==0.10.0` before PEFT;
- pins `transformers==5.13.1` and `peft==0.19.1` in the setup cell;
- releases CUDA cache between isolated method/seed runs;
- contains a final pass/fail check that Online-SDFT beats all baselines on both
  published metrics.

After changing embedded experiment code, rebuild and test:

```bash
python build_standalone_notebook.py
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q
```

Do not manually create a second notebook implementation that can drift from
the package.

## Website

The live GitHub Pages site is generated from `BLOG.md`; the Markdown file is
the single source of truth for the narrative:

- `website/build_site.py`: renders `BLOG.md` into `index.html` (protects math
  from the Markdown converter, wraps figures with captions, promotes Mermaid
  blocks, rewrites repository links, and builds the hero and contents list);
- `website/styles.css`: academic project-page layout — narrow reading column
  with figures that extend past it;
- `website/assets/og.png`: social-preview card;
- `website/index.html`: build output, committed so the site can be previewed
  without running the builder.

Math uses `$...$` and `$$...$$` in `BLOG.md` so it renders both on GitHub and,
via KaTeX, on the site. Mermaid and KaTeX load from CDN; nothing else is
bundled. There are no interactive modules right now — the plan is to add them
back one section at a time on top of this render.

The user prefers this article style over a product landing page. Preserve:

- narrow reading column, wide figures, restrained ink/teal palette;
- the story order: online continual learning first, bandit formulation second;
- semantic headings, alt text, and reduced-motion handling;
- precise, professional copy without playful metaphors.

Rebuild after editing `BLOG.md` or the builder:

```bash
python3 -m pip install markdown
python3 website/build_site.py
```

The workflow runs the same builder and copies these existing figures into
`_site`:

- `figures/blog_teaser.png`

- `figures/online_sdft_process.gif`
- `figures/bandit_accuracy.png`
- `figures/bandit_learning_curves.png`

Pushes touching `website/**`, `figures/**`, or the workflow trigger deployment.
Verify the new run and then verify the live URL returns the new content.

## Validation commands

Quick checks:

```bash
python3 website/build_site.py
git diff --check
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q
```

`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` avoids an unrelated globally installed
`torchtyping` pytest plugin that is incompatible with this machine's PyTorch.
It is not a repository failure.

Full experiment:

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python run.py --device cuda
```

Use `--device mps`, `--device cpu`, or `--device cuda` as appropriate. The
first run downloads the Liquid checkpoint from Hugging Face.

## Git and deployment workflow

Normal handoff sequence:

```bash
git status --short
git diff --check
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q
git add <scoped files>
git commit -m "..."
git push origin main
```

Then inspect the latest workflow under:

<https://github.com/lin826/Online-SFT-Demo/actions/workflows/pages.yml>

and verify:

<https://lin826.github.io/Online-SFT-Demo/>

## Local files to ignore

The local workspace may contain `.venv/`, `.DS_Store`, `.pytest_cache/`,
`__pycache__/`, `.claude/`, and ignored large output artifacts. These are not
part of the tracked project. Do not stage them accidentally.

## Recommended reading order for a new agent

1. `HANDOFF.md`
2. `README.md`
3. `docs/problem-setting.md`
4. `docs/methods.md`
5. `docs/evaluation.md`
6. `online_sdft/environment.py`
7. `online_sdft/methods.py`
8. `online_sdft/experiment.py`
9. `tests/`
10. `BLOG.md` and `website/build_site.py`

When changing behavior, update the package, notebook builder/notebook, docs,
tests, and website only where the same claim or implementation is duplicated.
Preserve the causal information boundary above all else.
