# Results

## Setup

Two runs, identical architecture (97.24M params: 12 layers, 12 heads, 768-dim,
512-token context) and identical data (FineWeb-Edu pretraining corpus,
smol-smoltalk SFT), differing only in the optimizer/normalization:

| | `baseline_v18` | `final_v24` |
|---|---|---|
| Optimizer | AdamW only | Muon (2D hidden weights) + AdamW (embeddings, norms) |
| QK-norm | no | yes |
| Pretrain iterations | 14,000 | 14,000 |
| SFT iterations | 4,500 | 4,500 |

Both trained on Kaggle's free 2x T4 GPU quota, effective batch size 192 x 512
tokens.

## Pretraining: loss vs. iteration (FineWeb-Edu validation set)

| Iteration | baseline val loss | final_v24 val loss |
|---|---|---|
| 4,000 | 3.5462 | 3.4018 |
| 5,000 | 3.5187 | 3.3446 |
| 6,500 | 3.4547 | 3.2989 |
| 8,000 | 3.3797 | 3.2834 |

Muon+QK-norm consistently ran ~0.10-0.17 nats below the AdamW-only baseline
at matched iteration counts — roughly a 30-40% head start in "iterations to
reach a given loss" in this range. (Both runs completed all 14,000
iterations; the table above covers the range with retained per-step logs.)

## SFT: final result (both runs, full 4,500 iterations)

| | baseline_v18 | final_v24 |
|---|---|---|
| Final val loss | 1.6932 | **1.6696** |
| Perplexity | 5.44 | **5.31** |

A modest, real improvement (~2.3% lower perplexity) — much smaller than an
earlier, invalidated run suggested (see *Development notes* below for why
that number couldn't be trusted). The pretraining-phase advantage narrows by
the end of SFT, most likely because 4,500 iterations on a comparatively small
148M-token chat corpus is enough for both runs to converge close to what this
model size and dataset can support.

## Benchmark suite (`chat/eval_benchmarks.py`, `final_v24`, n=500 per task)

Standard multiple-choice log-likelihood scoring — the same method
[lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)
uses (score every answer choice by the model's log-probability, pick the
argmax; no generation, no LLM-as-judge). Re-implemented from scratch here
since GPT-light isn't a HuggingFace model class.

| Task | acc | acc_norm | random baseline |
|---|---|---|---|
| ARC-Easy | 42.8% | 42.4% | 25% |
| ARC-Challenge | 18.4% | 23.0% | 25% |
| HellaSwag | 33.4% | 38.6% | 25% |
| LAMBADA (last-word) | 22.0% | -- | ~0% |

Clearly above chance on the easier tasks, at or below chance on ARC-Challenge
(genuinely hard multi-step reasoning questions). This sits below published
numbers for similarly-sized production models like GPT-2-124M or
SmolLM2-135M — those trained on 20-2000x more tokens. The gap is a data/compute
budget difference, not an architecture difference: GPT-light already uses
RoPE, SwiGLU, RMSNorm, and QK-norm, which is a more modern stack than
original GPT-2. See the root [README](README.md) for the fuller comparison.

*(Single run per task, no averaging over multiple seeds or prompt
formats — treat these as a snapshot, not a tight confidence interval.)*

## Example completions (`final_v24`, plain continuation, no chat template)

```
Prompt: The history of the Roman Empire begins with
Completion: the rise of the Roman Republic in the 4th century BCE. The Roman
Empire was founded by Emperor Augustus in 427 CE, and the Empire was founded
in 427 CE by Emperor Augustus.

Prompt: Photosynthesis is the process by which plants
Completion: convert sunlight energy into chemical energy. It's the process
by which plants absorb light and convert carbon dioxide into oxygen.

Prompt: The largest planet in our solar system is
Completion: Europa, with a surface area of approximately 3.5 million square
kilometers. The surface area of Europa is approximately 15.4% that of Earth.
```

Fluent, grammatical, topically on-target -- and confidently wrong on
specifics (dates, the planet itself, the repeated self-contradicting date).
That combination is typical at this scale: the model has learned what a
correct-sounding answer *looks like* far more reliably than it has learned
the underlying facts.

## Development notes

A few things that went wrong during training, kept here because they were
non-obvious and cost real GPU quota to find:

- **Silent data mix-up.** The pretraining data loader picked the first
  `train.bin` it found via a glob pattern; when both the FineWeb-Edu and the
  SFT dataset were attached to the same Kaggle kernel, mount order was not
  guaranteed, and one full session ended up "pretraining" on the much
  smaller, much easier SFT chat corpus instead. Fixed by excluding SFT
  datasets from the pretrain glob explicitly. The run this produced looked
  like a huge win (lower loss than it had any right to have) and had to be
  discarded once the mix-up was found.
- **AMP scaler state wasn't checkpointed.** Every resumed session restarted
  PyTorch's gradient scaler at its default value instead of wherever it had
  adapted to, causing a rough transition after each resume. Fixed by saving
  and restoring `GradScaler.state_dict()` alongside the model and optimizer
  state.
- **A stale iteration number in SFT checkpoints.** When pretraining finished
  and SFT started within the same session, the SFT checkpoint saver kept
  writing the pretrain-phase iteration counter instead of the (correct,
  completed) final value -- harmless as long as a session never got
  interrupted mid-SFT-after-pretrain, but wrong when one did, and a >1-hour
  resume would have silently re-run pretraining over the SFT weights.
- **GPU accelerator auto-assignment.** Kaggle occasionally assigned a P100
  instead of the expected 2x T4; the code's CUDA availability check caught
  the resulting driver mismatch and fell back to CPU *silently*, turning a
  training session into several wasted hours before anyone noticed the
  throughput. Fixed by pinning `machine_shape` explicitly in
  `kernel-metadata.json`.
