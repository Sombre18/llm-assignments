"""
04_transformer_block.py
=======================
Assembling attention into a full Transformer block.

A single block contains four components, in order:
  1. Layer normalisation  (stabilise inputs before attention)
  2. Multi-head attention (tokens communicate)
  3. Layer normalisation  (stabilise before feedforward)
  4. Feedforward network  (each token thinks independently)

Each of attention and feedforward is wrapped in a residual connection.
"""

import torch
import torch.nn as nn
from attention import MultiHeadAttention


# ---------------------------------------------------------------------------
# Feedforward network
# ---------------------------------------------------------------------------

class FeedForward(nn.Module):
    """
    A small two-layer MLP applied independently to each token position.

    After attention lets tokens communicate, the feedforward network
    gives each token a chance to process what it has gathered —
    on its own, without looking at neighbours.

    The expansion factor (typically 4x) creates a wider intermediate
    layer where the model can represent more complex combinations
    before projecting back down. Think of it as a working memory
    for each position.

    GELU activation is smoother than ReLU and works better in practice
    for language models — it doesn't hard-zero negative values,
    allowing small gradients to flow through.
    """

    def __init__(self, embed_dim: int, expansion: int = 4, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, expansion * embed_dim),
            nn.GELU(),
            nn.Linear(expansion * embed_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# Residual connections and layer normalisation
# ---------------------------------------------------------------------------
#
# Residual connection: output = x + sublayer(x)
#
# Instead of replacing x, each sublayer adds a correction to it.
# This has a profound effect on training: gradients flow directly
# from the output back to early layers without passing through
# every transformation. Deep networks become trainable.
#
# Think of it as: the model starts as an identity function and
# learns to make small adjustments, rather than learning everything
# from scratch.
#
# Layer normalisation: normalises across the embed_dim dimension
# (not across the batch). This keeps activations at a stable scale
# throughout training, preventing both vanishing and exploding gradients.
#
# We use "pre-norm" (normalise before each sublayer), which is more
# stable than the original "post-norm" formulation.


# ---------------------------------------------------------------------------
# The full block
# ---------------------------------------------------------------------------

class TransformerBlock(nn.Module):
    """
    One Transformer decoder block: attention + feedforward, both with
    residual connections and pre-layer-normalisation.

    Stacking N of these gives you a GPT.
    """

    def __init__(self, embed_dim: int, num_heads: int, expansion: int = 4, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn  = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ff    = FeedForward(embed_dim, expansion, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-norm residual: normalise → sublayer → add back to original
        x = x + self.attn(self.norm1(x))
        x = x + self.ff(self.norm2(x))
        return x


# ---------------------------------------------------------------------------
# The residual stream view (an important mental model)
# ---------------------------------------------------------------------------
#
# Rather than thinking of each layer as "transforming" the representation,
# think of a shared residual stream that flows through the network.
# Each block reads from it and writes small updates back.
#
# Implication: early layers and late layers all have direct access
# to the original token embeddings via the residual path.
# The network doesn't have to "remember" the input — it's always there.


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    B, T, C = 2, 16, 64
    num_heads = 4

    block = TransformerBlock(embed_dim=C, num_heads=num_heads)
    x = torch.randn(B, T, C)
    out = block(x)

    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {out.shape}")
    print(f"\nShape is preserved through the block.")
    print(f"The block has {sum(p.numel() for p in block.parameters()):,} parameters.")

    # Show that the residual keeps the signal alive
    # If attention/ff were zeroed out, output would equal input
    with torch.no_grad():
        for p in block.attn.parameters():
            p.zero_()
        for p in block.ff.parameters():
            p.zero_()
        out_zeroed = block(x)
        diff = (out_zeroed - x).abs().max().item()
        print(f"\nWith zeroed sublayers, max |output - input| = {diff:.6f}")
        print("(near zero — the residual connection preserves the input)")
