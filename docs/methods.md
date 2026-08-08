# Compared Methods

[← Main README](../README.md)

All six methods face the same paired event streams. Five use the highest LFM
action-token probability with 6% uniform exploration. REINFORCE samples from
its LFM policy because its gradient requires on-policy actions. No method sees
current feedback or the evaluator's preferred action before committing.

| Method | Adaptation mechanism |
| --- | --- |
| Base | Frozen Liquid LFM2.5-230M |
| ICL | Last 12 sampled teacher actions enter the LFM prompt; weights stay frozen |
| RAG | 12 nearest past teacher actions enter the LFM prompt; weights stay frozen |
| REINFORCE | Batch-one LoRA policy-gradient update from factual scalar reward only |
| Online-SFT | Updates a LoRA adapter from one sampled, one-hot teacher action |
| **Online-SDFT** | Updates the same LoRA adapter from the teacher's complete soft action distribution |

## Code boundaries

| Module | Owns | Cannot do |
| --- | --- | --- |
| `environment.py` | Stream, hidden `z`, factual outcomes, reward, teacher, scoring utility | Update the LFM |
| `methods.py` | Liquid policy, memory/retrieval, SFT/SDFT targets, LoRA update | Access `Event.z`, reward simulation, or oracle utilities |
| `experiment.py` | Predict → score → execute → teach → update ordering | Define rewards or learning algorithms |
| `reporting.py` | Aggregation, examples, figures | Affect any live action |

The bridge gives each method a `StudentObservation(text, features)`, never the
full simulator `Event`.

## What the LLM does

The deployed policy is the actual
[`LiquidAI/LFM2.5-230M`](https://huggingface.co/LiquidAI/LFM2.5-230M)
causal language model, not a linear proxy. Each prompt contains the currently
visible notification fields and asks for one single-token code:
`A = INTERRUPT`, `B = LATER`, or `C = ARCHIVE`. The softmax over those three
next-token logits is the student's action distribution.

REINFORCE, Online-SFT, and Online-SDFT train a rank-4 LoRA adapter with 172,032
trainable parameters. The underlying 230M-parameter LFM stays frozen. Base,
ICL, and RAG use the same LFM and identically initialized adapter but do not
update weights.

The LLM is the deployed student. For a controlled, auditable benchmark, the
privileged teacher is an explicit stochastic simulator policy rather than a
second LLM. It sees permitted post-decision state and factual feedback, but not
the evaluator's oracle action or utility vector.

## ICL and RAG, exactly

Both are genuine frozen-LFM baselines. After round `t` is over, they append the
legal pair `(visible notification x_t, sampled teacher route y_t)` to memory.
They never store the evaluator's best route. At round `t+1`, each selected pair
is formatted exactly like the live query:

```text
example 1 notification: category=...; hour=...; ...
example 1 route: B
...
current notification: category=...; hour=...; ...
current route:
```

The next-token probabilities of `A/B/C` are the policy; neither baseline makes
a gradient update. This is standard few-shot ICL: input-label demonstrations
condition a frozen LM. Example selection matters substantially in ICL, so RAG
uses nearest past demonstrations rather than an intentionally weak random
sample ([Liu et al., 2022](https://aclanthology.org/2022.deelio-1.10/);
[Rubin et al., 2022](https://aclanthology.org/2022.naacl-main.191/)).

| Detail | ICL | RAG |
| --- | --- | --- |
| Search pool | Last 12 legal records | Every legal past record |
| Selection | Recency | Top 12 nearest contexts |
| Prompt budget | 12 examples | 12 examples |
| Order | Oldest → newest | Weaker → stronger match, so the best is next to the query |
| Parameters | Frozen | Frozen |

RAG uses an equal-weight, Gower-style score over the six fields visible in the
notification: exact category match, exact regime match, proximity in
importance/deadline/affinity, and circular proximity in hour. This is a strong
zero-training retriever for mixed structured data: 23:30 is close to 00:30,
and a constant bias feature cannot swamp useful distinctions. It uses no hidden
`z`, feedback from the current round, future record, reward, or oracle utility.
We do not pretrain a learned retriever because that would add side labeled data
absent from the declared protocol. Training one from past records would instead
define a separate online-learning method, not this frozen RAG baseline.

Authoritative implementation: [`LiquidLLMPolicy.render_prompt`](../online_sdft/methods.py),
[`ICLAgent`](../online_sdft/methods.py),
[`mixed_context_similarity`](../online_sdft/methods.py), and
[`RAGAgent`](../online_sdft/methods.py). Their causal insertion point is
[`run_method`](../online_sdft/experiment.py): act first, append memory only
after execution and teacher feedback.

## REINFORCE, exactly

REINFORCE is the reward-only contextual-bandit baseline. It does not query or
store the teacher. At round `t`, it samples from the LFM action distribution,
executes that one route, and receives only its factual scalar reward `r_t`.
A past-only exponential moving average `b_t` reduces variance:

```text
a_t ~ student(. | x_t)
r_t = execute_only(a_t).reward
A_t = r_t - b_t

loss = -A_t log student(a_t | x_t) - 0.01 * entropy(student(. | x_t))
update_lora([(x_t, a_t, A_t)])        # batch size 1; affects t+1
b_(t+1) = b_t + 0.05 * (r_t - b_t)
```

There is no replay, reward-to-go, counterfactual reward, teacher distribution,
or oracle action. Because this is a one-step contextual bandit, the observed
reward is the complete return. The entropy term discourages immediate collapse;
the causal baseline never includes the current reward before computing the
advantage. [`REINFORCEAgent`](../online_sdft/methods.py) owns the baseline and
[`LiquidLLMPolicy.reinforce_update`](../online_sdft/methods.py) owns the
selected-action log-probability loss.

## Online-SDFT contract

The implementation enforces the three defining requirements:

1. **Own rollout:** the action comes from an epsilon-greedy policy over the
   student's own `student(. | x_t)` scores, without privileged `z_t`.
2. **Teacher supervision:** only after execution does `teacher(. | x_t, a_t, z_t)` produce a target. The simulator's oracle action is never a target.
3. **Small online updates:** the fresh record plus at most three recent replay
   records update the policy for `t+1`.

```text
a_t = epsilon_greedy(student(x_t, past_records), epsilon=0.06)
z_t = execute_only(a_t)
q_t = teacher(x_t, a_t, z_t)

replay.append((x_t, q_t))
batch = fresh_record + sample(up_to_3_recent_records)
student.update_lora(batch)             # affects only t+1 onward
```

## Online-SFT versus Online-SDFT

Both methods use the same post-decision teacher. Their only supervision difference is the target retained from that teacher:

| | Online-SFT | Online-SDFT |
| --- | --- | --- |
| Target | One sampled teacher action | Full teacher distribution `q_t` |
| Information retained | Hard winner only | Relative preference among all routes |
| Target variance | Higher | Lower |
| Update timing | After each factual interaction | After each factual interaction |
| Batch | Fresh + up to three recent records | Fresh + up to three recent records |

SDFT does not train on ground-truth demonstrations. Its advantage comes from preserving the teacher's uncertainty and action ranking instead of reducing them to one noisy sampled action.

The authoritative code is deliberately separated:
[`environment.py`](../online_sdft/environment.py) owns the privileged teacher,
[`methods.py`](../online_sdft/methods.py) owns the LFM and all six agents, and
[`experiment.py`](../online_sdft/experiment.py) owns the chronological
interaction loop.

See [Problem setting](problem-setting.md) for the causal order and [Results](results.md) for the comparison.
