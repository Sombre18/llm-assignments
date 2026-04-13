"""
02_embeddings.py
================
Turning tokens into vectors, and telling the model where each token sits.

Two distinct problems:
  1. Identity: what is this token?  → token embedding
  2. Position: where is this token? → positional embedding

Both are necessary. Without positional information, the model sees a
*set* of tokens, not a *sequence*. "Dog bites man" and "Man bites dog"
would be identical inputs. That would be bad.
"""

import torch
import torch.nn as nn
import math


# ---------------------------------------------------------------------------
# Token embeddings
# ---------------------------------------------------------------------------

class TokenEmbedding(nn.Module):
    """
    A lookup table: integer token ID → dense vector of floats.

    Think of it as a dictionary where each word (token) maps to a
    point in a high-dimensional space. Similar tokens end up near
    each other in that space — not because we told them to,
    but because gradient descent pushes them there during training.

    vocab_size:  number of distinct tokens (characters in our case)
    embed_dim:   length of each vector — the model's hidden dimension
    """

    def __init__(self, vocab_size: int, embed_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        # token_ids: [batch, seq_len]
        # returns:   [batch, seq_len, embed_dim]
        return self.embedding(token_ids)


# ---------------------------------------------------------------------------
# Positional embeddings — three options
# ---------------------------------------------------------------------------

class LearnedPositionalEmbedding(nn.Module):
    """
    One learned vector per position, up to max_seq_len.
    Simple, effective, and the default choice for most GPT-style models.
    Limitation: cannot generalise beyond max_seq_len seen during training.
    """

    def __init__(self, max_seq_len: int, embed_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(max_seq_len, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, seq_len, embed_dim]
        seq_len = x.shape[1]
        positions = torch.arange(seq_len, device=x.device)
        return x + self.embedding(positions)


class SinusoidalPositionalEncoding(nn.Module):
    """
    Fixed (non-learned) encoding from the original "Attention Is All You Need".

    Each position gets a unique pattern of sines and cosines at different
    frequencies. The intuition: low-frequency waves encode coarse position
    (am I near the start or end?), high-frequency waves encode fine position
    (which exact token am I?).

    Advantage over learned: works at inference time on sequences longer
    than anything seen during training.
    """

    def __init__(self, max_seq_len: int, embed_dim: int):
        super().__init__()
        pe = torch.zeros(max_seq_len, embed_dim)
        position = torch.arange(max_seq_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # register_buffer: not a parameter (won't be updated by optimizer),
        # but moves to GPU with the model automatically.
        self.register_buffer("pe", pe.unsqueeze(0))  # [1, max_seq_len, embed_dim]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.shape[1]
        return x + self.pe[:, :seq_len]


# ---------------------------------------------------------------------------
# Combined input embedding (what the model actually uses)
# ---------------------------------------------------------------------------

class InputEmbedding(nn.Module):
    """
    Token embedding + positional embedding, with dropout for regularisation.

    Dropout randomly zeros some values during training, forcing the model
    not to rely too heavily on any single feature. Think of it as
    deliberately introducing noise so the model learns robust patterns.
    """

    def __init__(self, vocab_size: int, embed_dim: int, max_seq_len: int, dropout: float = 0.1):
        super().__init__()
        self.token = TokenEmbedding(vocab_size, embed_dim)
        self.position = LearnedPositionalEmbedding(max_seq_len, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.embed_dim = embed_dim

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        # Scale token embeddings by sqrt(embed_dim) — a standard trick
        # that keeps the token and positional signals at similar magnitudes
        # when embed_dim is large.
        x = self.token(token_ids) * math.sqrt(self.embed_dim)
        x = self.position(x)
        return self.dropout(x)


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    vocab_size = 50
    embed_dim = 16
    max_seq_len = 32
    batch_size = 2
    seq_len = 10

    emb = InputEmbedding(vocab_size, embed_dim, max_seq_len)

    token_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    output = emb(token_ids)

    print(f"Input shape (token IDs):  {token_ids.shape}")
    print(f"Output shape (embeddings): {output.shape}")
    print(f"  = [batch={batch_size}, seq_len={seq_len}, embed_dim={embed_dim}]")
    print(f"\nEach token is now a vector of {embed_dim} floats.")
    print(f"These vectors will be refined by every subsequent layer.")
