# Compared Methods

[← Main README](../README.md)

All five methods face the same paired event streams. Each normally executes the
route with the highest LFM action-token probability and uses 6% uniform
exploration. No method sees current feedback or the evaluator's preferred action
before committing.

| Method | Adaptation mechanism |
| --- | --- |
| Base | Frozen Liquid LFM2.5-230M |
| ICL | Recent sampled teacher actions enter the LFM prompt; weights stay frozen |
| RAG | Similar past teacher actions enter the LFM prompt; weights stay frozen |
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

Online-SFT and Online-SDFT train a rank-4 LoRA adapter with 172,032 trainable
parameters. The underlying 230M-parameter LFM stays frozen. Base, ICL, and RAG
use the same LFM and identically initialized adapter but do not update weights.

The LLM is the deployed student. For a controlled, auditable benchmark, the
privileged teacher is an explicit stochastic simulator policy rather than a
second LLM. It sees permitted post-decision state and factual feedback, but not
the evaluator's oracle action or utility vector.

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
[`methods.py`](../online_sdft/methods.py) owns the LFM and all five agents, and
[`experiment.py`](../online_sdft/experiment.py) owns the chronological
interaction loop.

See [Problem setting](problem-setting.md) for the causal order and [Results](results.md) for the comparison.
