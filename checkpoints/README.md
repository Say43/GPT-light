# Checkpoints

Model weights (`checkpoint.pt`, ~800MB–1.1GB each) are **not** in this git
repository — GitHub's file size limits make that impractical, and PyTorch
checkpoints don't diff or compress usefully anyway. This folder keeps the
directory structure and the raw Kaggle training logs; the weights themselves
live locally and as private Kaggle datasets.

## What's here

- **`baseline_v18/`** — plain AdamW, no QK-norm, no Muon. The control group
  for the optimizer/normalization comparison described in the root
  [README](../README.md) and [RESULTS.md](../RESULTS.md).
- **`final_v24/`** — the final model: QK-norm + Muon, full pretrain (14,000
  iterations on FineWeb-Edu) + SFT (4,500 iterations on smol-smoltalk). This
  is what `chat/local_chat.py` and `chat/eval_benchmarks.py` load by default.

Each folder's `training.log` is the raw stdout from the Kaggle run that
produced it — training-loss curves, bpb values, and the example completions
Kaggle printed at the end.

## Reproducing a checkpoint

There's no public download link for these specific weights (the Kaggle
datasets are private to my account). To get a comparable checkpoint:

1. Prepare a pretokenized FineWeb-Edu corpus and upload it as a Kaggle
   dataset (`train.bin`/`val.bin`/`meta.json`, `uint16` token ids — see
   `kaggle/pretrain_sft.ipynb` cell 3 for the exact format expected).
2. Push `kaggle/` with the [Kaggle CLI](https://github.com/Kaggle/kaggle-api):
   `kaggle kernels push -p kaggle`.
3. Kaggle's free GPU quota caps a session at 12h and ~30h/week, so a full
   14,000-iteration pretrain run needs 2-3 sessions. After each session,
   download the output checkpoint, re-upload it as a new Kaggle dataset, and
   add it to `dataset_sources` in `kernel-metadata.json` before the next
   push — the notebook resumes from it automatically (optimizer, Muon, and
   AMP-scaler state all included).
