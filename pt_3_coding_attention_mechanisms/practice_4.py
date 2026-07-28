# Task: Building a Multi-Head Attention Mechanism

import torch

class MultiHeadAttention(torch.nn.Module):

    def __init__(self, embedding_dim: int, num_heads: int, dropout: float, context_length: int, qkv_bias: bool = False) -> None:
        super().__init__()

        assert embedding_dim % num_heads == 0

        self.num_heads = num_heads
        self.head_dim = int(embedding_dim / num_heads)

        self.query_proj = torch.nn.Linear(in_features=embedding_dim, out_features=embedding_dim, bias=qkv_bias)
        self.key_proj = torch.nn.Linear(in_features=embedding_dim, out_features=embedding_dim, bias=qkv_bias)
        self.value_proj = torch.nn.Linear(in_features=embedding_dim, out_features=embedding_dim, bias=qkv_bias)
        self.output_proj = torch.nn.Linear(in_features=embedding_dim, out_features=embedding_dim, bias=qkv_bias)

        self.dropout = torch.nn.Dropout(dropout)
        self.register_buffer("mask", torch.triu(torch.ones(context_length, context_length, dtype=torch.bool), diagonal=1))

    def split_to_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq, embedding_dim = x.shape
        x = x.view(batch, seq, self.num_heads, self.head_dim)
        return x.transpose(1, 2)

    def forward(self, x):

        # Make queries, keys, values
        queries = self.query_proj(x)
        keys = self.key_proj(x)
        values = self.value_proj(x)

        # Split queries, keys, values into heads
        queries = self.split_to_heads(x=queries)
        keys = self.split_to_heads(x=keys)
        values = self.split_to_heads(x=values)

        # Make attention scores
        attention_scores = queries @ keys.transpose(-2, -1)

        # Mask attention scores to prevent model from seeing future words
        seq_len = queries.shape[-2]
        attention_scores = attention_scores.masked_fill(self.mask.bool()[:seq_len, :seq_len], -torch.inf)

        # Normalize attention scores to weights
        attention_weights = torch.softmax((attention_scores / self.head_dim**0.5), dim=-1)
        attention_weights = self.dropout(attention_weights)

        context_vec = attention_weights @ values

        batch, num_heads, seq, head_dim = context_vec.shape
        context_vec = context_vec.transpose(1, 2).contiguous()
        context_vec = context_vec.view(batch, seq, num_heads * head_dim)

        return self.output_proj(context_vec)


x = MultiHeadAttention(512, 8, 0.1, 10)
input = torch.rand(size=(16, 10, 512))
print(x(input).shape)