"""
03_attention.py
===============
The attention mechanism: how tokens communicate with each other.

The core idea: when processing a token, the model should be able to
"look back" at previous tokens and selectively pull in relevant information.
Not all context is equally useful — attention learns which parts matter.

The library analogy
-------------------
Imagine you're writing a word and need to consult the context so far.
  Query (Q): your search request — "what do I need right now?"
  Key   (K): the index card for each past token — "what do I offer?"
  Value (V): the actual content each past token carries.

You compare your query against all keys (dot product = relevance score),
normalise the scores into weights (softmax), then take a weighted sum
of the values. The result: a context-aware representation of your position.

The cognitive science angle: this is not unlike how human attention works —
selective, content-driven, and operating in parallel across a field.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ---------------------------------------------------------------------------
# Single-head attention (the building block)
# ---------------------------------------------------------------------------

class SingleHeadAttention(nn.Module):
    """
    One attention head: Q, K, V projections + scaled dot-product attention.

    embed_dim:  input/output dimension
    head_dim:   dimension of Q, K, V vectors (often embed_dim // num_heads)
    """

    def __init__(self, embed_dim: int, head_dim: int, dropout: float = 0.1):
        super().__init__()
        # Three separate linear projections — no bias by convention in modern GPTs
        self.q_proj = nn.Linear(embed_dim, head_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, head_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, head_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, seq_len, embed_dim]
        Q = self.q_proj(x)  # [batch, seq_len, head_dim]
        K = self.k_proj(x)  # [batch, seq_len, head_dim]
        V = self.v_proj(x)  # [batch, seq_len, head_dim]

        # Use PyTorch's optimised implementation (uses FlashAttention if available)
        # is_causal=True applies the causal mask: position t cannot attend to t+1, t+2, ...
        # This is what makes the model autoregressive — it can only see the past.
        out = F.scaled_dot_product_attention(Q, K, V, dropout_p=0.0, is_causal=True)
        return out  # [batch, seq_len, head_dim]


# ---------------------------------------------------------------------------
# Why multiple heads?
# ---------------------------------------------------------------------------
#
# One head can only learn one kind of relationship at a time.
# With multiple heads running in parallel, the model can simultaneously track:
#   - syntactic structure (which word governs which)
#   - semantic similarity (which words mean related things)
#   - positional patterns (rhyme at line endings, meter)
#
# Each head gets a slice of the embedding dimension (embed_dim // num_heads),
# so total compute stays the same as one full-dimension head.
# The outputs are concatenated and projected back to embed_dim.


class MultiHeadAttention(nn.Module):
    """
    Multi-head causal self-attention.

    embed_dim must be divisible by num_heads.
    Each head operates on embed_dim // num_heads dimensions.
    """

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.embed_dim = embed_dim

        # Project input into Q, K, V for all heads at once (efficient)
        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim, bias=False)
        # Project concatenated heads back to embed_dim
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.dropout = dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape  # batch, sequence length, embed_dim

        # Compute Q, K, V for all heads in one matrix multiply, then split
        qkv = self.qkv_proj(x)                          # [B, T, 3*C]
        Q, K, V = qkv.split(self.embed_dim, dim=-1)     # each [B, T, C]

        # Reshape to expose the head dimension
        # [B, T, C] → [B, T, num_heads, head_dim] → [B, num_heads, T, head_dim]
        def reshape_for_heads(t):
            return t.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        Q, K, V = reshape_for_heads(Q), reshape_for_heads(K), reshape_for_heads(V)

        # Scaled dot-product attention across all heads simultaneously
        # is_causal=True handles the causal mask internally (no future tokens)
        attn_out = F.scaled_dot_product_attention(
            Q, K, V,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True
        )  # [B, num_heads, T, head_dim]

        # Reassemble heads: [B, num_heads, T, head_dim] → [B, T, C]
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, C)

        return self.out_proj(attn_out)  # [B, T, C]


# ---------------------------------------------------------------------------
# What does "scaled" mean in scaled dot-product attention?
# ---------------------------------------------------------------------------
#
# The dot product Q·K^T grows in magnitude with head_dim.
# Large values push softmax into regions where gradients vanish
# (the distribution becomes nearly one-hot, and learning stalls).
# Dividing by sqrt(head_dim) keeps the scale stable regardless of dimension.
# This is the "scaled" in scaled dot-product attention.


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    B, T, C = 2, 16, 64   # batch=2, seq_len=16, embed_dim=64
    num_heads = 4

    x = torch.randn(B, T, C)

    mha = MultiHeadAttention(embed_dim=C, num_heads=num_heads)
    out = mha(x)

    print(f"Input shape:  {x.shape}   [batch, seq_len, embed_dim]")
    print(f"Output shape: {out.shape}  [batch, seq_len, embed_dim]")
    print(f"\nAttention is shape-preserving: input and output dimensions match.")
    print(f"Each position now carries information from all previous positions.")
    print(f"\nhead_dim = {C} // {num_heads} = {C // num_heads}")
    print(f"Each of {num_heads} heads attends independently, then results are merged.")
