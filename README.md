# GPT-light

Training a GPT language model **from scratch** — implemented in plain PyTorch, small enough to pretrain on free-tier cloud GPUs (Google Colab / Kaggle T4), but built like the real thing.

The project started as an extension of the Karpathy GPT learning workflow: instead of stopping at a character-level toy model, it moves step by step toward a realistic small language model — tokenized data, a proper pretraining corpus, and multi-session training on free GPU quotas. It is deliberately not trying to reproduce a production model; the goal is to bridge the gap between a teaching implementation and a real assistant-style training pipeline.

## What's inside

The model is a decoder-only transformer written from first principles (no `transformers` model classes — only the tokenizer is reused):

- ~124M parameters: 12 layers, 12 heads, 768-dim embeddings, 512-token context
- RMSNorm, causal self-attention, learned positional embeddings
- Cosine learning-rate schedule with warmup, gradient accumulation (effective batch of 192 × 512 tokens), AdamW
- Checkpoint save/resume so training can span multiple free GPU sessions
- Sampling with temperature and top-k for text-completion tests

## Training pipeline (phased)

1. **Phase 1–2** — architecture bring-up and smoke tests on small chat data (`GPT-light-colab.ipynb`)
2. **Phase 3** — pretraining on a pretokenized [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) corpus, uploaded as a Kaggle dataset so GPU quota isn't wasted on data prep (`kaggle_notebook/`)
3. **Phase 4 (planned)** — supervised fine-tuning on `smol-smoltalk` to turn the base model into an assistant

## Repository layout

```text
GPT-light-colab.ipynb        Original Colab training notebook
kaggle_notebook/
  notebook09c30ece6c.ipynb   Kaggle training notebook (GPU, checkpoint resume)
  kernel-metadata.json       Kaggle kernel config (datasets, GPU settings)
  output_v*/                  Training run logs
```

## Running it

On **Kaggle**: push the notebook with the [Kaggle CLI](https://github.com/Kaggle/kaggle-api) (`kaggle kernels push -p kaggle_notebook`), with the pretokenized dataset and checkpoint dataset attached. The notebook resumes from the latest checkpoint automatically.

On **Colab**: open `GPT-light-colab.ipynb` and run top to bottom on a GPU runtime.

## Notes

- The pretrained result is a **base model** (text continuation), not a chat assistant — instruction following arrives with the SFT phase.
- Batch size and gradient accumulation are tuned to fit a 16 GB T4; the logs in `output_v*` document the OOM experiments that led there.
