# GPT-light

**A GPT language model built and trained from scratch** — the model itself is
written from first principles in plain PyTorch, not assembled from a
pre-existing model library, and its weights were trained from random
initialization on public text data. No pretrained weights are downloaded and
adapted; the only component reused from elsewhere is the tokenizer.

The result is a 97-million-parameter model that writes fluent, grammatical
English, answers questions in a chat format, and scores clearly above chance
on standard reading-comprehension benchmarks — trained end to end within the
free GPU quota of a public cloud notebook platform.

**Results, benchmark numbers, and an honest account of what went wrong:
see [RESULTS.md](RESULTS.md).**


## Motivation

The project began as a follow-on to Karpathy's *"let's build GPT"* style
tutorials. Those typically stop at a character-level toy model that
demonstrates the mechanism but not the engineering around it. The goal here
was to carry the same construction all the way to a realistic small-scale
training pipeline, and to find out empirically how much of that pipeline
survives a low compute budget:

- a tokenized pretraining corpus at scale rather than a single text file,
- a training run spanning multiple sessions and weekly GPU-quota resets,
- an architectural change validated by a controlled comparison rather than
  assumed to help,
- and quantitative evaluation on established benchmarks rather than
  eyeballing a handful of sample outputs.

It is not an attempt to compete with production models. It is an attempt to
reproduce the method at small scale and report the outcome accurately.

## Architecture

Decoder-only transformer, 97.24M parameters: 12 layers, 12 attention heads,
768-dimensional embeddings, 512-token context window. The component choices
follow current efficient small-model practice rather than the 2017
Transformer defaults:

- **RoPE** (rotary position embeddings) instead of learned positional
  embeddings
- **SwiGLU** feedforward instead of a plain ReLU/GELU MLP
- **RMSNorm** with pre-norm residual blocks
- **QK-norm** — normalizing queries and keys before attention, which
  stabilizes training at higher learning rates
- Tied input/output embeddings
- **Muon** optimizer for the 2D hidden weight matrices (attention and
  feedforward projections), **AdamW** for embeddings and normalization gains.
  This is an independent reimplementation of the algorithm described by
  [Jordan et al.](https://kellerjordan.github.io/posts/muon/) — written from
  the published description, not copied from the (MIT-licensed) reference
  code.
- **WSD** (warmup–stable–decay) learning-rate schedule, chosen because it does
  not require committing to a total step count in advance — a practical
  necessity when sessions are terminated by quota limits and the remaining
  budget for the week is unknown.

These choices track what recent efficient small-model projects use, most
directly [Karpathy's nanochat](https://github.com/karpathy/nanochat), which
served as a reference point for QK-norm and Muon in particular.

## Training pipeline

1. **Prototype** (`notebooks/01_colab_prototype.ipynb`) — architecture
   bring-up and smoke tests on a small chat dataset in Google Colab.
2. **Pretraining** — 14,000 iterations on a pretokenized
   [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu)
   corpus (990M tokens). The corpus was tokenized offline and uploaded as a
   dataset so that training sessions spend GPU time on training rather than
   on data preparation.
3. **Supervised fine-tuning (SFT)** — 4,500 iterations continuing from the
   pretrained checkpoint, on
   [smol-smoltalk](https://huggingface.co/datasets/HuggingFaceTB/smol-smoltalk)
   conversations in a `<|user|>`/`<|assistant|>` format. The loss is masked to
   the assistant turns only (`ignore_index=-100` on user tokens), so the model
   learns to answer questions rather than to generate them.

Both phases run in the same notebook (`kaggle/pretrain_sft.ipynb`) and share a
checkpoint-and-resume mechanism, since a complete run does not fit into a
single session. Model weights, the state of both optimizers, the mixed-
precision gradient-scaler state, and the current phase and iteration are all
persisted together, so an interrupted run resumes at exactly the point it
stopped.

## Method and validation

The central experiment is a **single-variable comparison**: two runs with
identical architecture, identical data, and identical iteration counts,
differing only in the optimizer and the use of QK-norm. The AdamW-only run
(`baseline_v18`) serves as the control; `final_v24` is the modified
configuration. Measured effect: consistently lower validation loss during
pretraining, converging to a smaller but real ~2.3% improvement in perplexity
after fine-tuning. Full numbers are in [RESULTS.md](RESULTS.md).

Final model quality is measured with the standard multiple-choice
log-likelihood protocol on ARC-Easy, ARC-Challenge, HellaSwag, and LAMBADA —
the same scoring method used by
[lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness),
reimplemented here because GPT-light is not a HuggingFace model class.

[RESULTS.md](RESULTS.md) also documents the failures: a silent dataset
mix-up that invalidated an entire run and made the results look better than
they were, an unsaved scaler state that disturbed every resume, a stale
iteration counter that could have overwritten fine-tuned weights, and a
silent CPU fallback that wasted several hours of a session. They are recorded
because negative results are part of the measurement, and because each one
cost real GPU quota to find.

## Repository layout

```text
notebooks/01_colab_prototype.ipynb   Early architecture smoke-test (Colab)
kaggle/
  pretrain_sft.ipynb                 Main training notebook (pretrain + SFT, checkpoint resume)
  kernel-metadata.json               Kaggle kernel config (datasets, GPU pinning)
chat/
  local_chat.py                      Local CLI chat against a trained checkpoint
  eval_benchmarks.py                 Log-likelihood benchmark suite (ARC/HellaSwag/LAMBADA)
  tokenizer/                         Tokenizer files (shared by both scripts)
checkpoints/
  baseline_v18/                      AdamW-only control run (weights gitignored, see checkpoints/README.md)
  final_v24/                         Final model: QK-norm + Muon, full pretrain + SFT
logs/legacy/                         Early OOM-tuning debug logs from architecture bring-up
RESULTS.md                           Benchmark numbers, controlled comparison, and failure notes
```

## Running it

**Chat with the trained model locally:**
```
pip install -r requirements.txt
python chat/local_chat.py
```
Loads `checkpoints/final_v24/checkpoint.pt` by default. Each message starts a
fresh conversation (no multi-turn memory): at this model size, carrying prior
turns in the context window pulled replies off-topic more often than it
helped — see [RESULTS.md](RESULTS.md).

**Reproduce the benchmark scores:**
```
python chat/eval_benchmarks.py
```
Downloads ARC-Easy, ARC-Challenge, HellaSwag, and LAMBADA from Hugging Face
and scores the checkpoint with the log-likelihood method described above.

**Pretraining/SFT on Kaggle:** push the kernel with the
[Kaggle CLI](https://github.com/Kaggle/kaggle-api):
`kaggle kernels push -p kaggle`, with the pretokenized dataset attached (and a
checkpoint dataset attached when resuming — see
[checkpoints/README.md](checkpoints/README.md)). The notebook detects an
existing checkpoint and resumes automatically.

**Prototype in Colab:** open `notebooks/01_colab_prototype.ipynb` and run it
top to bottom on a GPU runtime.

## Scope and limitations

- The pretraining-only checkpoint is a **base model** — it continues text, it
  does not follow instructions. Instruction-following is acquired in the SFT
  phase.
- Benchmark scores fall below published numbers for similarly sized
  production models (GPT-2-124M, SmolLM2-135M). Those models were trained on
  20–2000x more tokens; the gap is a data and compute budget difference, not
  an architectural one — the stack used here (RoPE, SwiGLU, RMSNorm, QK-norm)
  is more modern than original GPT-2's.
- Benchmark results are a single run per task, without averaging over seeds or
  prompt formats. They should be read as a snapshot, not as a tight confidence
  interval.
- Batch size, gradient accumulation, and GPU pinning are tuned specifically
  around a 16GB T4; `logs/legacy/` documents the out-of-memory experiments
  that led to those settings.
- This is an educational and research-practice project, not a production
  system.
