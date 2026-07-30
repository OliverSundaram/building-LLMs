import torch
from torch import nn

class MultiHeadAttention(nn.Module):
    def __init__(self, cfg: dict[str, int | bool]):
        super().__init__()

        self.vocab_size = cfg["vocab_size"]
        self.context_length = cfg["context_length"]
        self.emb_dim = cfg["emb_dim"]
        self.n_heads = cfg["n_heads"]
        self.head_dim = int(self.emb_dim / self.n_heads)
        self.n_layers = cfg["n_layers"]
        self.drop_rate = cfg["drop_rate"]
        self.qkv_bias = cfg["qkv_bias"]

        assert self.emb_dim % self.n_heads == 0

        self.query_proj = nn.Linear(self.emb_dim, self.emb_dim, self.qkv_bias)
        self.key_proj = nn.Linear(self.emb_dim, self.emb_dim, self.qkv_bias)
        self.value_proj = nn.Linear(self.emb_dim, self.emb_dim, self.qkv_bias)
        self.out_proj = nn.Linear(self.emb_dim, self.emb_dim, self.qkv_bias)

        self.dropout = nn.Dropout(self.drop_rate)
        self.register_buffer("mask", torch.triu(torch.ones(self.context_length, self.context_length, dtype=torch.bool), diagonal=1))

    def split_to_heads(self, x: torch.Tensor):
        batch, seq, emb_dim = x.shape
        x = x.view(batch, seq, self.n_heads, self.head_dim)
        return x.transpose(1, 2)

    def forward(self, x):

        queries = self.query_proj(x)
        keys = self.key_proj(x)
        values = self.value_proj(x)

        queries = self.split_to_heads(queries)
        keys = self.split_to_heads(keys)
        values = self.split_to_heads(values)

        atten_scores = queries @ keys.transpose(2, 3)
        atten_scores /= self.head_dim**0.5

        seq_len = queries.shape[-2]
        atten_scores = atten_scores.masked_fill(self.mask.bool()[:seq_len, :seq_len], -torch.inf)

        atten_weights = torch.softmax(atten_scores, dim=-1)
        atten_weights = self.dropout(atten_weights)

        context_vec = atten_weights @ values

        batch, n_heads, seq, emb_dim = context_vec.shape
        context_vec = context_vec.transpose(1, 2).contiguous()
        context_vec = context_vec.view(batch, seq, self.emb_dim)

        return self.out_proj(context_vec)



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



class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            nn.GELU(),
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

    def forward(self, x):

        shortcut = x
        x = self.norm1(x)
        x = self.att(x)
        x += shortcut

        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x += shortcut

        return x



class GPT2Model(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])

        self.transformer_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )

        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.linear_out = nn.Linear(cfg["emb_dim"], 2)
        self.dropout = nn.Dropout(cfg["drop_rate"])

    def forward(self, x: torch.Tensor):
        batch, seq = x.shape
        tok_emb = self.tok_emb(x)
        pos_emb = self.pos_emb(torch.arange(seq, device=x.device))

        return self.linear_out(self.final_norm(self.transformer_blocks(self.dropout(tok_emb + pos_emb))))