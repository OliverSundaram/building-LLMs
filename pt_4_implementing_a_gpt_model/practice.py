import torch
from torch import nn

import tiktoken
tokenizer = tiktoken.get_encoding("gpt2")

class MultiHeadAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        assert cfg["emb_dim"] % cfg["n_heads"] == 0

        self.num_heads = cfg["n_heads"]
        self.head_dim = int(cfg["emb_dim"] / cfg["n_heads"])

        emb_dim = cfg["emb_dim"]
        qkv_bias = cfg["qkv_bias"]

        self.query_proj = nn.Linear(emb_dim, emb_dim, qkv_bias)
        self.key_proj = nn.Linear(emb_dim, emb_dim, qkv_bias)
        self.value_proj = nn.Linear(emb_dim, emb_dim, qkv_bias)
        self.output_proj = nn.Linear(emb_dim, emb_dim, qkv_bias)

        self.dropout = nn.Dropout(cfg["drop_rate"])
        self.register_buffer("mask", torch.triu(torch.ones(cfg["context_length"], cfg["context_length"], dtype=torch.bool), diagonal=1))

    def split_to_heads(self, x):
        batch, seq, emb_dim = x.shape
        x = x.view(batch, seq, self.num_heads, self.head_dim)
        return x.transpose(1, 2)

    def forward(self, x):

        queries = self.query_proj(x)
        keys = self.key_proj(x)
        values = self.value_proj(x)

        queries = self.split_to_heads(queries)
        keys = self.split_to_heads(keys)
        values = self.split_to_heads(values)

        attention_scores = queries @ keys.transpose(-2, -1)

        seq_len = queries.shape[-2]
        attention_scores = attention_scores.masked_fill(self.mask.bool()[:seq_len, :seq_len], -torch.inf)

        attention_weights = torch.softmax((attention_scores / self.head_dim**0.5), dim=-1)
        attention_weights = self.dropout(attention_weights)

        context_vec = attention_weights @ values

        batch, num_heads, seq, head_dim = context_vec.shape
        context_vec = context_vec.transpose(1, 2).contiguous()
        context_vec = context_vec.view(batch, seq, num_heads * head_dim)

        return self.output_proj(context_vec)

class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(torch.sqrt(torch.tensor(2.0 / torch.pi)) * (x + 0.044715 * torch.pow(x, 3))))

class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"])
        )

    def forward(self, x):
        return self.layers(x)

class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention(cfg)
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):

        shortcut = x
        x = self.drop_shortcut(self.att(self.norm1(x)))
        x += shortcut

        shortcut = x
        x = self.drop_shortcut(self.ff(self.norm2(x)))
        x += shortcut

        return x

class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift

class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        self.transformer_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )

        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, x: torch.Tensor):
        batch_size, seq_len = x.shape
        tok_emb = self.tok_emb(x)
        pos_emb = self.pos_emb(torch.arange(seq_len, device=x.device))

        return self.out_head(self.final_norm(self.transformer_blocks(self.drop_emb(tok_emb + pos_emb))))

def generate_text(model, text, max_new_tokens, context_size):

    idx = tokenizer.encode(text)
    idx = torch.tensor(idx).unsqueeze(0)

    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]

        with torch.inference_mode():
            logits = model(idx_cond)

        logits = logits[:, -1]
        probs = torch.softmax(logits, dim=-1)
        idx_next = torch.argmax(probs, dim=-1, keepdim=True)
        idx = torch.cat((idx, idx_next), dim=1)

    idx = idx.squeeze()
    idx = idx.tolist()
    text = tokenizer.decode(idx)

    return text

GPT_CONFIG = {
    "vocab_size": 50257,
    "context_length": 1024,
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": False
}

start_context = "Hello, I am"
model = GPTModel(GPT_CONFIG)

out = generate_text(
    model=model,
    text=start_context,
    max_new_tokens=6,
    context_size=GPT_CONFIG["context_length"]
)

print(out)