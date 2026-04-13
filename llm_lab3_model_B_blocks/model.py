"""
05_model.py
===========
The full character-level GPT: assembling all previous modules.

Architecture:
  input token IDs
      ↓
  InputEmbedding   (token + positional)
      ↓
  N × TransformerBlock
      ↓
  LayerNorm        (final normalisation)
      ↓
  Linear           (project to vocab_size — the "language model head")
      ↓
  logits over vocabulary  →  sample next character

The output dimension is vocab_size at every sequence position.
During training, every position predicts its next token simultaneously.
During generation, only the last position matters.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from embeddings import InputEmbedding
from transformer_block import TransformerBlock


class CharGPT(nn.Module):
    """
    Character-level GPT decoder.

    Parameters
    ----------
    vocab_size    : number of distinct characters in corpus
    embed_dim     : hidden dimension throughout the model
    num_heads     : attention heads per block (embed_dim must be divisible)
    num_layers    : number of stacked TransformerBlocks
    context_length: maximum sequence length the model can process
    dropout       : dropout rate applied throughout
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 6,
        context_length: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.context_length = context_length

        self.embedding = InputEmbedding(vocab_size, embed_dim, context_length, dropout)

        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, expansion=4, dropout=dropout)
            for _ in range(num_layers)
        ])

        self.final_norm = nn.LayerNorm(embed_dim)

        # The language model head: embed_dim → vocab_size
        # No softmax here — we return raw logits.
        # The loss function (cross_entropy) and the sampler handle the rest.
        self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False)

        # Weight tying: share weights between the token embedding table
        # and the output projection. This is a well-established trick —
        # the embedding that maps token→vector and the head that maps
        # vector→token are learning the same space, so tying them
        # reduces parameters and often improves performance.
        self.lm_head.weight = self.embedding.token.embedding.weight

        self._init_weights()

    def _init_weights(self):
        """
        Small initialisations prevent activations from exploding before
        training begins. The 0.02 std is standard for GPT-style models.
        Residual projections are scaled down further (by 1/sqrt(num_layers))
        so the residual stream starts near identity.
        """
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        token_ids: [batch, seq_len]   (seq_len ≤ context_length)
        returns:   [batch, seq_len, vocab_size]  — logits at every position
        """
        x = self.embedding(token_ids)       # [B, T, embed_dim]

        for block in self.blocks:
            x = block(x)                    # [B, T, embed_dim]

        x = self.final_norm(x)              # [B, T, embed_dim]
        logits = self.lm_head(x)            # [B, T, vocab_size]
        return logits

    @torch.no_grad()
    def generate(self, prompt_ids: torch.Tensor, max_new_tokens: int,
                 temperature: float = 1.0, top_k: int = None) -> torch.Tensor:
        """
        Autoregressive generation: predict one token at a time,
        append it, repeat.

        temperature: > 1.0 = more random/creative, < 1.0 = more conservative.
                     At 0 this approaches greedy (always pick the top token).
        top_k:       if set, sample only from the top-k most likely tokens.
                     Prevents very unlikely characters from ever appearing.

        Note the asymmetry with training:
          Training   — all positions compute loss in parallel (efficient).
          Generation — strictly sequential; each step needs the previous output.
        """
        self.eval()
        ids = prompt_ids.clone()

        for _ in range(max_new_tokens):
            # Crop context to the model's window if needed
            context = ids[:, -self.context_length:]

            logits = self(context)          # [B, T, vocab_size]
            logits = logits[:, -1, :]      # only the last position: [B, vocab_size]

            logits = logits / temperature

            if top_k is not None:
                # Zero out all logits except the top-k
                values, _ = torch.topk(logits, top_k)
                threshold = values[:, -1].unsqueeze(-1)
                logits = logits.masked_fill(logits < threshold, float('-inf'))

            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)  # [B, 1]
            ids = torch.cat([ids, next_id], dim=1)

        return ids

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    vocab_size = 65   # typical for a Shakespeare-sized character set
    model = CharGPT(
        vocab_size=vocab_size,
        embed_dim=256,
        num_heads=8,
        num_layers=6,
        context_length=256,
    )

    print(f"Model parameters: {model.count_parameters():,}")
    print()

    # Forward pass
    B, T = 2, 64
    token_ids = torch.randint(0, vocab_size, (B, T))
    logits = model(token_ids)
    print(f"Input:  {token_ids.shape}  [batch, seq_len]")
    print(f"Output: {logits.shape}  [batch, seq_len, vocab_size]")
    print(f"  Every position predicts a distribution over {vocab_size} characters.")
    print()

    # Generation
    prompt = torch.zeros(1, 1, dtype=torch.long)   # single "start" token
    generated = model.generate(prompt, max_new_tokens=100, temperature=1.0, top_k=40)
    print(f"Generated sequence length: {generated.shape[1]} tokens")
    print("(random weights → random output, but the pipeline works)")
