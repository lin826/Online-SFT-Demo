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

### Exact selection and serving logic

```text
# Shared causal memory insertion, after the action and feedback:
memory.append((visible_x_t, sample(teacher(. | x_t, a_t, z_t))))

# ICL at t+1:
examples = memory[-12:]

# RAG at t+1:
score(record, current) = mean(
    same_category,
    same_regime,
    1 - abs(delta_importance),
    1 - abs(delta_deadline),
    1 - abs(delta_affinity),
    1 - circular_hour_distance / 12_hours,
)
examples = top_12(memory, key=score)
examples = order_weaker_to_stronger_match(examples)

llm_probs = softmax(next_token_logits(prompt(examples, current))[A, B, C])
behavior = 0.94 * one_hot(argmax(llm_probs)) + 0.06 / 3
action = sample(behavior)
```

The record's factual feedback remains available for auditing, but the ICL/RAG
prompt uses only its visible notification and sampled teacher route. RAG never
uses the route label when calculating similarity. Both agents have their own
causal histories because their earlier actions can produce different feedback
and therefore different teacher samples.

### Worked prompts and actual LFM responses

This is a reconstructed decision from the published MPS trace: **seed 0,
decision 148**. Both agents see the same current notification:

```text
category=social; hour=17.2; regime=on-call; importance=0.28;
deadline=0.10; affinity=0.83
```

The Liquid chat template adds its control tokens around this shared system
message:

```text
You are an on-device notification router.
Choose exactly one route and reply with only its code:
A = INTERRUPT now
B = LATER in a digest
C = ARCHIVE without a notification
Use the current notification and any past teacher examples. Do not add explanation.
```

ICL takes source decisions `136–147`, regardless of similarity. Its 12 labels
are `7×A`, `1×B`, and `4×C`.

<details>
<summary><strong>Expand the complete ICL user prompt</strong></summary>

```text
example 1 notification: category=teammate; hour=13.9; regime=on-call; importance=0.38; deadline=0.20; affinity=0.41
example 1 route: C
example 2 notification: category=teammate; hour=13.1; regime=on-call; importance=0.60; deadline=0.32; affinity=0.59
example 2 route: C
example 3 notification: category=manager; hour=12.8; regime=on-call; importance=0.84; deadline=0.68; affinity=0.32
example 3 route: C
example 4 notification: category=calendar; hour=16.0; regime=on-call; importance=0.96; deadline=0.98; affinity=0.17
example 4 route: A
example 5 notification: category=manager; hour=16.2; regime=on-call; importance=0.65; deadline=0.65; affinity=0.55
example 5 route: A
example 6 notification: category=social; hour=11.9; regime=on-call; importance=0.22; deadline=0.13; affinity=0.93
example 6 route: A
example 7 notification: category=manager; hour=13.8; regime=on-call; importance=0.63; deadline=0.72; affinity=0.34
example 7 route: B
example 8 notification: category=manager; hour=13.3; regime=on-call; importance=0.81; deadline=0.64; affinity=0.50
example 8 route: A
example 9 notification: category=manager; hour=16.1; regime=on-call; importance=0.75; deadline=0.81; affinity=0.84
example 9 route: A
example 10 notification: category=teammate; hour=19.5; regime=on-call; importance=0.32; deadline=0.07; affinity=0.44
example 10 route: C
example 11 notification: category=calendar; hour=15.0; regime=on-call; importance=0.79; deadline=0.96; affinity=0.48
example 11 route: A
example 12 notification: category=monitoring; hour=15.8; regime=on-call; importance=0.86; deadline=0.85; affinity=0.46
example 12 route: A
Treat these as user-specific evidence, not universal rules.
current notification: category=social; hour=17.2; regime=on-call; importance=0.28; deadline=0.10; affinity=0.83
current route:
```

</details>

The LFM response is the single token `A`:

```text
assistant: A
```

`P(A)=0.5127`, `P(B)=0.4041`, `P(C)=0.0832`; therefore ICL's greedy and
executed route is `INTERRUPT`.

RAG searches all 147 legal past records. It retrieves source decisions
`129, 95, 135, 117, 133, 141, 94, 122, 104, 111, 112, 134`, ordered from
similarity `0.7471` to `0.9449` so the best match is closest to the query.

<details>
<summary><strong>Expand the complete RAG user prompt</strong></summary>

```text
example 1 notification: category=teammate; hour=15.5; regime=on-call; importance=0.42; deadline=0.24; affinity=0.73
example 1 route: B
example 2 notification: category=teammate; hour=16.5; regime=on-call; importance=0.43; deadline=0.09; affinity=0.62
example 2 route: C
example 3 notification: category=receipt; hour=17.6; regime=on-call; importance=0.29; deadline=0.09; affinity=0.54
example 3 route: A
example 4 notification: category=social; hour=13.9; regime=on-call; importance=0.00; deadline=0.11; affinity=0.63
example 4 route: C
example 5 notification: category=social; hour=14.4; regime=on-call; importance=0.00; deadline=0.00; affinity=0.76
example 5 route: B
example 6 notification: category=social; hour=11.9; regime=on-call; importance=0.22; deadline=0.13; affinity=0.93
example 6 route: C
example 7 notification: category=social; hour=14.4; regime=on-call; importance=0.03; deadline=0.16; affinity=0.88
example 7 route: C
example 8 notification: category=social; hour=15.6; regime=on-call; importance=0.12; deadline=0.00; affinity=0.94
example 8 route: C
example 9 notification: category=social; hour=15.3; regime=on-call; importance=0.38; deadline=0.00; affinity=0.72
example 9 route: A
example 10 notification: category=social; hour=14.3; regime=on-call; importance=0.30; deadline=0.16; affinity=0.95
example 10 route: C
example 11 notification: category=social; hour=14.7; regime=on-call; importance=0.21; deadline=0.09; affinity=0.71
example 11 route: B
example 12 notification: category=social; hour=14.2; regime=on-call; importance=0.28; deadline=0.10; affinity=0.76
example 12 route: C
Treat these as user-specific evidence, not universal rules.
current notification: category=social; hour=17.2; regime=on-call; importance=0.28; deadline=0.10; affinity=0.83
current route:
```

</details>

The LFM response is the single token `B`:

```text
assistant: B
```

`P(A)=0.1182`, `P(B)=0.8136`, `P(C)=0.0682`; therefore RAG's greedy and
executed route is `LATER`. The evaluator also scores `LATER` best in this
round, but that fact was sealed away from both prompts and used only afterward
for online accuracy and regret.

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
