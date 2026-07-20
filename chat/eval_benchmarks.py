"""Broadband benchmark of the GPT-light model.

Uses the same log-likelihood multiple-choice scoring as EleutherAI's
lm-evaluation-harness (github.com/EleutherAI/lm-evaluation-harness): for each
question, score every answer choice by the total log-probability the model
assigns to that continuation, then pick the argmax. No text generation and no
LLM-as-judge -- this is the standard, deterministic way tiny models are graded.

Two accuracies per task, exactly as the harness reports them:
  acc      -- argmax of summed log-prob of the continuation
  acc_norm -- argmax of log-prob normalized by continuation byte length
              (removes the bias toward shorter answers)

Datasets pulled live from HuggingFace: allenai/ai2_arc (ARC-Easy),
Rowan/hellaswag, ybisk/piqa, EleutherAI/lambada_openai.
"""
import os
import sys
import math
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import local_chat as lc  # reuses the exact model + tokenizer + checkpoint

model = lc.raw_model
tok = lc.tokenizer
device = lc.device
block_size = lc.block_size

N_PER_TASK = int(os.environ.get('N_PER_TASK', '500'))  # subsample for speed


@torch.no_grad()
def continuation_logprob(context_ids, cont_ids):
    """Sum of log p(cont | context), token by token. lm-eval 'loglikelihood'."""
    ids = (context_ids + cont_ids)[-block_size:]
    # how many of the trailing tokens belong to the continuation (after clip)
    n_cont = min(len(cont_ids), len(ids) - 1)
    if n_cont <= 0:
        return -1e30
    x = torch.tensor([ids[:-1]], dtype=torch.long, device=device)
    logits, _ = model(x)
    logprobs = F.log_softmax(logits[0].float(), dim=-1)
    # targets are ids[1:]; we only score the last n_cont positions
    targets = torch.tensor(ids[1:], dtype=torch.long, device=device)
    tok_lp = logprobs[torch.arange(len(targets)), targets]
    return tok_lp[-n_cont:].sum().item()


def score_mc(items):
    """items: list of (context_str, [choice_str,...], gold_idx). Returns (acc, acc_norm, n)."""
    correct = correct_norm = 0
    for ctx, choices, gold in items:
        ctx_ids = tok.encode(ctx)
        lps, lps_norm = [], []
        for ch in choices:
            cont_ids = tok.encode(ch)
            lp = continuation_logprob(ctx_ids, cont_ids)
            lps.append(lp)
            lps_norm.append(lp / max(len(ch), 1))  # per-byte normalization
        if max(range(len(lps)), key=lambda i: lps[i]) == gold:
            correct += 1
        if max(range(len(lps_norm)), key=lambda i: lps_norm[i]) == gold:
            correct_norm += 1
    n = len(items)
    return correct / n, correct_norm / n, n


def _load_arc(config):
    from datasets import load_dataset
    ds = load_dataset('allenai/ai2_arc', config, split='test')
    items = []
    for ex in ds.select(range(min(N_PER_TASK, len(ds)))):
        labels = ex['choices']['label']
        texts = ex['choices']['text']
        if ex['answerKey'] not in labels:
            continue
        gold = labels.index(ex['answerKey'])
        ctx = f"Question: {ex['question']}\nAnswer:"
        choices = [f" {t}" for t in texts]
        items.append((ctx, choices, gold))
    return items


def load_arc_easy():
    return _load_arc('ARC-Easy')


def load_arc_challenge():
    return _load_arc('ARC-Challenge')


def load_hellaswag():
    from datasets import load_dataset
    ds = load_dataset('Rowan/hellaswag', split='validation')
    items = []
    for ex in ds.select(range(min(N_PER_TASK, len(ds)))):
        ctx = ex['ctx']
        choices = [' ' + e for e in ex['endings']]
        gold = int(ex['label'])
        items.append((ctx, choices, gold))
    return items


@torch.no_grad()
def eval_lambada():
    """Last-word prediction accuracy (LAMBADA). Standard tiny-model LM probe.
    Tokenizes context+target jointly so the continuation uses its natural
    in-context tokenization (the separate-encode boundary was too strict)."""
    from datasets import load_dataset
    ds = load_dataset('EleutherAI/lambada_openai', 'en', split='test')
    correct = 0
    n = 0
    for ex in ds.select(range(min(N_PER_TASK, len(ds)))):
        text = ex['text'].strip()
        cut = text.rfind(' ')
        if cut <= 0:
            continue
        ctx = text[:cut]  # target = the last whitespace-delimited word
        full_ids = tok.encode(text)
        ctx_ids = tok.encode(ctx)
        n_last = len(full_ids) - len(ctx_ids)
        ids = full_ids[-block_size:]
        n_last = min(n_last, len(ids) - 1)
        if n_last <= 0:
            continue
        x = torch.tensor([ids[:-1]], dtype=torch.long, device=device)
        logits, _ = model(x)
        pred = logits[0, -n_last:].argmax(dim=-1)
        targets = torch.tensor(ids[-n_last:], dtype=torch.long, device=device)
        if torch.equal(pred, targets):
            correct += 1
        n += 1
    return correct / n, n


TASKS = [
    ('ARC-Easy',  load_arc_easy,      0.25),
    ('ARC-Chall', load_arc_challenge, 0.25),
    ('HellaSwag', load_hellaswag,     0.25),
]

if __name__ == '__main__':
    print(f'\nModel: {lc.CKPT_PATH}')
    print(f'Samples per task: {N_PER_TASK}\n')
    print(f'{"Task":<12}{"acc":>8}{"acc_norm":>10}{"random":>9}{"n":>7}')
    print('-' * 46)
    for name, loader, rand in TASKS:
        try:
            items = loader()
            acc, acc_norm, n = score_mc(items)
            print(f'{name:<12}{acc*100:>7.1f}%{acc_norm*100:>9.1f}%{rand*100:>8.0f}%{n:>7}')
        except Exception as e:
            print(f'{name:<12} FAILED: {e}')
    try:
        lacc, ln = eval_lambada()
        print(f'{"LAMBADA":<12}{lacc*100:>7.1f}%{"--":>10}{"~0%":>9}{ln:>7}')
    except Exception as e:
        print(f'{"LAMBADA":<12} FAILED: {e}')
    print('\n(acc_norm is the headline metric for ARC/HellaSwag; LAMBADA is exact last-word match.)')
