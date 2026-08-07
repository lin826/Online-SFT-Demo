# Compared Methods

[← Main README](../README.md)

All five methods face the same paired event streams and act with 6% exploration. No method sees current feedback or the evaluator's preferred action before committing.

| Method | Adaptation mechanism |
| --- | --- |
| Base | Frozen generic linear policy |
| ICL | Recent sampled teacher actions remain in context; weights stay frozen |
| RAG | Retrieves sampled teacher actions from similar past contexts; weights stay frozen |
| Online-SFT | Updates weights from one sampled, one-hot teacher action |
| **Online-SDFT** | Updates weights from the teacher's complete soft action distribution |

## Online-SDFT contract

The implementation enforces the three defining requirements:

1. **Own rollout:** the student samples `a_t` from `student(. | x_t)` without privileged `z_t`.
2. **Teacher supervision:** only after execution does `teacher(. | x_t, a_t, z_t)` produce a target. The simulator's oracle action is never a target.
3. **Small online updates:** the fresh record plus at most three older replay records update the policy for `t+1`.

```text
a_t = sample(student(x_t, past_records))
z_t = execute_only(a_t)
q_t = teacher(x_t, a_t, z_t)

replay.append((x_t, q_t))
batch = fresh_record + sample(up_to_3_older_records)
student.update(batch)          # affects only t+1 onward
```

## Online-SFT versus Online-SDFT

Both methods use the same post-decision teacher. Their only supervision difference is the target retained from that teacher:

| | Online-SFT | Online-SDFT |
| --- | --- | --- |
| Target | One sampled teacher action | Full teacher distribution `q_t` |
| Information retained | Hard winner only | Relative preference among all routes |
| Target variance | Higher | Lower |
| Update timing | After each factual interaction | After each factual interaction |
| Batch | Fresh + up to three replay items | Fresh + up to three replay items |

SDFT does not train on ground-truth demonstrations. Its advantage comes from preserving the teacher's uncertainty and action ranking instead of reducing them to one noisy sampled action.

The authoritative code is [`teacher_policy`](../bandit_experiment.py#L180-L205) and the update branch inside [`run_method`](../bandit_experiment.py#L289-L326).

See [Problem setting](problem-setting.md) for the causal order and [Results](results.md) for the comparison.
