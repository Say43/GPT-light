import os
import re
import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers import PreTrainedTokenizerFast

device = 'cuda' if torch.cuda.is_available() else 'cpu'

HERE = os.path.dirname(os.path.abspath(__file__))
TOK_DIR = os.path.join(HERE, 'tokenizer')
CKPT_PATH = os.path.join(HERE, '..', 'checkpoints', 'final_v24', 'checkpoint.pt')

tokenizer = PreTrainedTokenizerFast.from_pretrained(TOK_DIR)
tokenizer.pad_token = '<|endoftext|>'
tokenizer.eos_token = '<|endoftext|>'
vocab_size = tokenizer.vocab_size

n_embd = 768
n_head = 12
n_layer = 12
dropout = 0.0  # eval mode, dropout has no effect anyway
block_size = 512


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return norm * self.weight


def precompute_rope(head_size, max_seq_len, base=10000.0):
    inv_freq = 1.0 / (base ** (torch.arange(0, head_size, 2).float() / head_size))
    t = torch.arange(max_seq_len).float()
    freqs = torch.outer(t, inv_freq)
    return torch.cos(freqs), torch.sin(freqs)


def apply_rope(x, cos, sin):
    x1, x2 = x[..., ::2], x[..., 1::2]
    cos = cos[None, None, :x.shape[2], :].to(x.dtype)
    sin = sin[None, None, :x.shape[2], :].to(x.dtype)
    rotated = torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
    return rotated.flatten(-2)


class SwiGLU(nn.Module):
    def __init__(self, n_embd, hidden_mult=4):
        super().__init__()
        hidden = int(2 / 3 * hidden_mult * n_embd)
        self.w1 = nn.Linear(n_embd, hidden, bias=False)
        self.w3 = nn.Linear(n_embd, hidden, bias=False)
        self.w2 = nn.Linear(hidden, n_embd, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


def qk_rms_norm(x, eps=1e-6):
    # parameter-free QK-norm; must match the training-time setting, which is
    # recorded in the checkpoint ('use_qk_norm') and applied after loading
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)


class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd, n_head, block_size):
        super().__init__()
        self.n_head = n_head
        self.head_size = n_embd // n_head
        self.qkv = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.proj = nn.Linear(n_embd, n_embd, bias=False)
        self.dropout = dropout
        self.resid_dropout = nn.Dropout(dropout)
        self.use_qk_norm = False
        cos, sin = precompute_rope(self.head_size, block_size)
        self.register_buffer('rope_cos', cos, persistent=False)
        self.register_buffer('rope_sin', sin, persistent=False)

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.split(C, dim=2)
        q = q.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        if self.use_qk_norm:
            q = qk_rms_norm(q)
            k = qk_rms_norm(k)
        q = apply_rope(q, self.rope_cos, self.rope_sin)
        k = apply_rope(k, self.rope_cos, self.rope_sin)
        out = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.proj(out))


class Block(nn.Module):
    def __init__(self, n_embd, n_head, block_size):
        super().__init__()
        self.sa = CausalSelfAttention(n_embd, n_head, block_size)
        self.ffwd = SwiGLU(n_embd)
        self.ln1 = RMSNorm(n_embd)
        self.ln2 = RMSNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class GPTLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head, block_size) for _ in range(n_layer)])
        self.ln_f = RMSNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding_table.weight
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, idx, targets=None):
        x = self.token_embedding_table(idx)
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            B, T, C = logits.shape
            loss = F.cross_entropy(logits.view(B * T, C), targets.view(B * T))
        return logits, loss


raw_model = GPTLanguageModel().to(device)
print(f'{sum(p.numel() for p in raw_model.parameters()) / 1e6:.2f}M parameters')

ckpt = torch.load(CKPT_PATH, map_location=device)
raw_model.load_state_dict(ckpt['model_state_dict'])
use_qk_norm = ckpt.get('use_qk_norm', False)
for blk in raw_model.blocks:
    blk.sa.use_qk_norm = use_qk_norm
print(f"Loaded checkpoint from iter {ckpt['iter']} (qk_norm={use_qk_norm})")
raw_model.eval()

EOT_ID = tokenizer.convert_tokens_to_ids('<|endoftext|>')
USER_ID = tokenizer.convert_tokens_to_ids('<|user|>')
ASSISTANT_ID = tokenizer.convert_tokens_to_ids('<|assistant|>')
SYSTEM_ID = tokenizer.convert_tokens_to_ids('<|system|>')
STOP_IDS = {EOT_ID, USER_ID, ASSISTANT_ID, SYSTEM_ID}


@torch.no_grad()
def chat_generate(prompt, max_new_tokens=200, temperature=0.5, top_k=40,
                  repetition_penalty=1.3):
    # temperature 0.5 (was 0.7): the test battery showed the same prompt
    # succeeding or derailing purely by sampling luck; less randomness means
    # the model stays on the high-probability answer templates it memorized.
    ids = tokenizer.encode(prompt)
    idx = torch.tensor([ids], dtype=torch.long, device=device)
    out = []
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -block_size:]
        logits, _ = raw_model(idx_cond)
        logits = logits[:, -1, :]
        # Repetition penalty: dampen logits of tokens already generated in this
        # reply, so the small model doesn't lock into its loop attractors.
        if repetition_penalty != 1.0 and out:
            seen = torch.tensor(sorted(set(out)), dtype=torch.long, device=device)
            picked = logits[0, seen]
            logits[0, seen] = torch.where(picked > 0, picked / repetition_penalty,
                                          picked * repetition_penalty)
        logits = logits / temperature
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float('inf')
        probs = F.softmax(logits, dim=-1)
        nxt = torch.multinomial(probs, num_samples=1)
        tok = nxt.item()
        if tok in STOP_IDS:
            break
        out.append(tok)
        idx = torch.cat((idx, nxt), dim=1)
    return tokenizer.decode(out).strip()


# Stateless chat: every message starts from a clean slate. The test battery
# showed that at 97M params, prior turns in the context drag new answers back
# to the old topic more often than they help -- auto-reset beats /reset
# discipline. (Trade-off: no follow-up questions across messages.)

# smol-smoltalk conversations often open with a system turn; anchoring on one
# keeps short/generic user messages (like a bare "hello") from drifting into
# the dataset's dominant coding-task distribution.
SYSTEM_PROMPT = 'You are a friendly and helpful assistant. Answer the user directly and conversationally.'


def build_prompt(user_msg):
    parts = [f'<|system|>\n{SYSTEM_PROMPT}']
    parts.append(f'<|user|>\n{user_msg}')
    parts.append('<|assistant|>\n')
    return '\n\n'.join(parts)


def trim_to_sentence(text):
    # When generation stops at the max_new_tokens cap instead of an
    # end-of-turn token, the reply breaks off mid-sentence (seen repeatedly
    # in the test battery). Cut back to the last complete sentence when that
    # loses less than half the reply -- and only THEN drop a now-dangling
    # list enumerator ("8."), since the cut itself is what tends to leave
    # one behind (raw tail "8. Stay curious and ke" -> cut -> "8.").
    cut = max(text.rfind('.'), text.rfind('!'), text.rfind('?'))
    if cut != -1 and (cut >= len(text) // 2 or len(text) - cut <= 80):
        text = text[:cut + 1]
    return re.sub(r'(?<=[.!?])\s+\d+[.)]\s*$', '', text).rstrip()


def chat(user_msg, **kw):
    return trim_to_sentence(chat_generate(build_prompt(user_msg), **kw))


if __name__ == '__main__':
    print('device:', device)
    print('=== GPT-light lokaler Chat ===')
    print('Jede Nachricht startet ein frisches Gespraech (auto-reset). '
          'Leere Eingabe oder Strg+C beendet.')
    while True:
        try:
            msg = input('Du: ')
        except (EOFError, KeyboardInterrupt):
            break
        if not msg.strip():
            break
        reply = chat(msg)
        print('Assistant:', reply)
