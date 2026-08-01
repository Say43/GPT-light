# Third-party notices and attribution

This file records the third-party data, algorithms, and software that
GPT-light builds on, together with the licence each is made available under.
It exists to satisfy the attribution requirements of those licences —
principally the Open Data Commons Attribution Licence, which governs the
pretraining corpus.

Nothing in this file affects the status of the original code in this
repository; it concerns only the third-party components listed below.

## 1. Training data

### FineWeb-Edu — attribution required

The pretraining corpus is
[FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu),
created by Hugging Face and released under the **Open Data Commons
Attribution License (ODC-By) v1.0**. Use of the dataset is additionally
subject to [Common Crawl's Terms of Use](https://commoncrawl.org/terms-of-use),
since FineWeb-Edu is derived from Common Crawl data.

Two artefacts in this repository are produced works derived from that
database, and this notice is the attribution that ODC-By requires for them:

- `chat/tokenizer/tokenizer.json` — the byte-level BPE vocabulary, trained
  during preparation of the FineWeb-Edu corpus.
- The trained model weights (`checkpoint.pt`). These are **not** distributed
  in this repository or anywhere publicly; see
  [checkpoints/README.md](checkpoints/README.md). Should they ever be
  published, this notice must accompany them.

Citation:

```bibtex
@misc{lozhkov2024fineweb-edu,
    author       = { Lozhkov, Anton and Ben Allal, Loubna and
                     von Werra, Leandro and Wolf, Thomas },
    title        = { FineWeb-Edu: the Finest Collection of Educational Content },
    year         = 2024,
    url          = { https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu },
    doi          = { 10.57967/hf/2497 },
    publisher    = { Hugging Face }
}
```

### smol-smoltalk

The supervised fine-tuning corpus is
[smol-smoltalk](https://huggingface.co/datasets/HuggingFaceTB/smol-smoltalk),
created by Hugging Face and released under the **Apache License 2.0**.

No content from either corpus is redistributed here. The committed notebooks
contain no saved cell outputs, and the training logs contain loss values and
model-generated text only.

## 2. Evaluation datasets

`chat/eval_benchmarks.py` downloads these at runtime from the Hugging Face
Hub. They are **not** redistributed in this repository; the licences are
recorded here for completeness, and the reported scores are measurements, not
excerpts of the data.

| Dataset | Source | Licence |
|---|---|---|
| ARC-Easy, ARC-Challenge | [`allenai/ai2_arc`](https://huggingface.co/datasets/allenai/ai2_arc) | CC BY-SA 4.0 |
| HellaSwag | [`Rowan/hellaswag`](https://huggingface.co/datasets/Rowan/hellaswag) | MIT |
| LAMBADA (OpenAI variant) | [`EleutherAI/lambada_openai`](https://huggingface.co/datasets/EleutherAI/lambada_openai) | Modified MIT |

## 3. Algorithms and reference implementations

- **Muon optimizer** — the algorithm is described by Keller Jordan et al.,
  ["Muon: An optimizer for hidden layers in neural networks"](https://kellerjordan.github.io/posts/muon/).
  The reference implementation is MIT-licensed. The implementation in this
  repository was written independently from the published description; no
  reference code was copied. The attribution here is scientific credit for
  the algorithm, which is not itself subject to copyright.
- **nanochat** — [karpathy/nanochat](https://github.com/karpathy/nanochat)
  (MIT) served as a reference point for architectural choices, in particular
  QK-norm and the use of Muon. No code was copied from it.
- The early prototype notebook (`notebooks/01_colab_prototype.ipynb`)
  downloads the `Qwen/Qwen-tokenizer` vocabulary at runtime for smoke tests.
  It is not redistributed here, and the training pipeline does not use it.

## 4. Software dependencies

Installed from their respective package indexes at runtime and not
redistributed in this repository:

| Package | Licence |
|---|---|
| PyTorch | BSD-3-Clause |
| NumPy | BSD-3-Clause |
| Hugging Face `transformers`, `datasets` | Apache License 2.0 |
| `ipywidgets` | BSD-3-Clause |

Training was carried out on Kaggle Notebooks under Kaggle's terms for
free GPU use.

## Corrections

If you believe an attribution here is incomplete or incorrect, please open an
issue — it will be fixed.
