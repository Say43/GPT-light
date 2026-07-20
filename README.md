# GPT-light

A GPT language model trained **from scratch** in plain PyTorch — no
`transformers` model classes, only the tokenizer is reused. Small enough to
pretrain end to end on free-tier cloud GPUs (Kaggle's 2x T4 quota), but built
with the same architectural ingredients modern open models use, not the
2017 Transformer defaults.

It started as a follow-on to Karpathy's "let's build GPT" style tutorials:
instead of stopping at a character-level toy model, it goes step by step
toward something closer to a real small-model training pipeline —
tokenized data at scale, a real pretraining corpus, multi-session training
across weekly GPU quota resets, an optimizer swap validated with an A/B
comparison, and a standard benchmark suite instead of just eyeballing a few
completions. It's not trying to compete with production models; the goal was
to see how much of that pipeline holds up under a hobbyist's compute budget.

**Results, benchmark numbers, and honest failure notes: see
[RESULTS.md](RESULTS.md).**

## Architecture

Decoder-only transformer, 97.24M parameters (12 layers, 12 heads, 768-dim
embeddings, 512-token context):

- **RoPE** (rotary position embeddings) instead of learned positional
  embeddings
- **SwiGLU** feedforward instead of a plain ReLU/GELU MLP
- **RMSNorm**, pre-norm residual blocks
- **QK-norm** — normalizing queries/keys before attention, for training
  stability at higher learning rates
- Tied input/output embeddings
- **Muon** optimizer for the 2D hidden weight matrices (attention and
  feedforward projections), **AdamW** for embeddings and norm gains — a
  from-scratch reimplementation of the algorithm described in
  [Jordan et al.](https://kellerjordan.github.io/posts/muon/) (MIT-licensed
  reference implementation; this is an independent implementation, not
  copied code)
- WSD (warmup-stable-decay) learning-rate schedule, so a training run isn't
  locked into a fixed total step count decided up front — useful when
  sessions get cut off by GPU-quota limits and you don't know in advance how
  many more you'll get this week

These choices track what recent efficient small-model projects use — most
directly [Karpathy's nanochat](https://github.com/karpathy/nanochat), which
was a reference point for several of them (QK-norm, Muon in particular).

## Training pipeline

1. **Prototype** (`notebooks/01_colab_prototype.ipynb`) — architecture
   bring-up and smoke tests on a small chat dataset in Google Colab.
2. **Pretraining** — a pretokenized [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu)
   corpus (990M tokens), prepared offline and uploaded as a Kaggle dataset so
   training sessions don't burn GPU quota on data prep. 14,000 iterations,
   WSD schedule.
3. **SFT** — continues from the pretrained checkpoint, fine-tuning on
   [smol-smoltalk](https://huggingface.co/datasets/HuggingFaceTB/smol-smoltalk)
   conversations in a `<|user|>`/`<|assistant|>` format. Loss is masked to
   only the assistant turns (`ignore_index=-100` on user tokens) so the model
   learns to answer, not to ask. 4,500 iterations.

Both phases run in the same notebook (`kaggle/pretrain_sft.ipynb`) and share
a checkpoint-resume mechanism, since a full run doesn't fit in one Kaggle
session: model weights, both optimizers' state, the AMP gradient-scaler
state, and which phase/iteration to resume from are all saved together.

## Repository layout

```text
notebooks/01_colab_prototype.ipynb   Early architecture smoke-test (Colab)
kaggle/
  pretrain_sft.ipynb                 Main training notebook (pretrain + SFT, checkpoint resume)
  kernel-metadata.json               Kaggle kernel config (datasets, GPU pinning)
chat/
  local_chat.py                      Local CLI chat against a trained checkpoint
  eval_benchmarks.py                 Standard log-likelihood benchmark suite (ARC/HellaSwag/LAMBADA)
  tokenizer/                         Tokenizer files (shared by both scripts)
checkpoints/
  baseline_v18/                      AdamW-only control run (weights gitignored, see checkpoints/README.md)
  final_v24/                         Final model: QK-norm + Muon, full pretrain + SFT
logs/legacy/                         Early OOM-tuning debug logs from architecture bring-up
RESULTS.md                           Benchmark numbers, A/B comparison, and what went wrong along the way
```

## Running it

**Pretraining/SFT on Kaggle:** push with the
[Kaggle CLI](https://github.com/Kaggle/kaggle-api):
`kaggle kernels push -p kaggle`, with the pretokenized dataset attached (and
a checkpoint dataset attached for resuming — see
[checkpoints/README.md](checkpoints/README.md)). The notebook resumes
automatically.

**Prototype in Colab:** open `notebooks/01_colab_prototype.ipynb` and run top
to bottom on a GPU runtime.

**Chat locally:**
```
pip install -r requirements.txt
python chat/local_chat.py
```
Loads `checkpoints/final_v24/checkpoint.pt` by default. Each message starts
a fresh conversation (no multi-turn memory) — at this model size, carrying
prior turns in context dragged replies off-topic more often than it helped;
see [RESULTS.md](RESULTS.md) for why.

**Run the benchmark suite:**
```
python chat/eval_benchmarks.py
```
Downloads ARC-Easy, ARC-Challenge, HellaSwag, and LAMBADA from Hugging Face
and scores the checkpoint with the standard log-likelihood method (see
[RESULTS.md](RESULTS.md) for results and methodology).

## What this is and isn't

- The pretrained-only checkpoint is a **base model** (text continuation),
  not an assistant — instruction-following comes from the SFT phase.
- It's a hobbyist / educational project, not a production model. Benchmark
  scores are below similarly-sized production small models (GPT-2-124M,
  SmolLM2-135M), because those trained on orders of magnitude more tokens.
  The architecture itself is arguably more modern than GPT-2's; the gap is
  purely a compute/data budget difference. Details in
  [RESULTS.md](RESULTS.md).
- Batch size, gradient accumulation, and GPU pinning are tuned around a
  16GB T4; `logs/legacy/` documents the OOM experiments that led there.
