"""
07_analysis.py
==============
Stage III & IV from the assignment: analysing what the model learned.

Two angles:
  A. Structural analysis of generated poems — does the output look like poetry?
     Line lengths, rhyme, stanza structure.

  B. Probing internal representations — does the model's hidden state
     encode linguistic structure, even though it was never explicitly taught?

Probing is the key idea: train a simple linear classifier on top of frozen
model activations. If a linear model can predict (say) "is this character
at the end of a line?" from the hidden state, then that information is
linearly encoded in the representation. The model has learned it implicitly.
"""

import torch
import torch.nn as nn
import numpy as np
from collections import Counter
import re


# ---------------------------------------------------------------------------
# A. Structural analysis of generated text
# ---------------------------------------------------------------------------

def line_lengths(text: str) -> list[int]:
    """Character count per line, excluding empty lines."""
    return [len(line) for line in text.split("\n") if line.strip()]


def stanza_lengths(text: str) -> list[int]:
    """Number of lines per stanza (stanzas separated by blank lines)."""
    stanzas = re.split(r"\n\s*\n", text.strip())
    return [len([l for l in s.split("\n") if l.strip()]) for s in stanzas]


def extract_line_endings(text: str, n_chars: int = 3) -> list[str]:
    """Last n characters of each non-empty line — where rhymes live."""
    endings = []
    for line in text.split("\n"):
        stripped = line.rstrip()
        if stripped:
            endings.append(stripped[-n_chars:].lower())
    return endings


def rhyme_density(text: str) -> float:
    """
    Fraction of adjacent line-ending pairs that share their last 2 characters.
    A rough proxy for rhyme. 0 = no rhyme, 1 = everything rhymes.
    """
    endings = extract_line_endings(text, n_chars=2)
    if len(endings) < 2:
        return 0.0
    matches = sum(a == b for a, b in zip(endings, endings[1:]))
    return matches / (len(endings) - 1)


def compare_corpora(train_text: str, generated_text: str):
    """
    Print a side-by-side comparison of structural statistics.
    The question: does the generated text resemble the training distribution?
    """
    def stats(lengths):
        a = np.array(lengths)
        return f"mean={a.mean():.1f}, std={a.std():.1f}, median={np.median(a):.1f}"

    print("=== Structural Comparison ===\n")
    print(f"Line length   | train: {stats(line_lengths(train_text))}")
    print(f"              | gen:   {stats(line_lengths(generated_text))}")
    print()
    print(f"Stanza length | train: {stats(stanza_lengths(train_text))}")
    print(f"              | gen:   {stats(stanza_lengths(generated_text))}")
    print()
    print(f"Rhyme density | train: {rhyme_density(train_text):.3f}")
    print(f"              | gen:   {rhyme_density(generated_text):.3f}")


# ---------------------------------------------------------------------------
# B. Probing classifiers
# ---------------------------------------------------------------------------
#
# Procedure:
#   1. Run the model on held-out text, recording hidden states at each layer.
#   2. For each position, define a label (e.g. "is this a newline character?").
#   3. Train a logistic regression on top of the frozen hidden states.
#   4. Accuracy above chance → the property is encoded in that layer.
#
# Why linear? If we used a deep MLP probe, it could learn the property
# itself rather than just reading it out. Linear probes test whether
# the information is *already there*, not whether it can be computed.


def extract_hidden_states(
    model: nn.Module,
    token_ids: torch.Tensor,
    layer_idx: int,
    device: str = "cpu",
) -> torch.Tensor:
    """
    Forward pass with a hook to capture the output of a specific block.

    Hooks are PyTorch's way of intercepting intermediate computations
    without modifying the model's forward method.

    Returns hidden states: [seq_len, embed_dim]
    """
    model.eval()
    hidden = {}

    def hook_fn(module, input, output):
        hidden["states"] = output.detach().cpu()

    handle = model.blocks[layer_idx].register_forward_hook(hook_fn)

    with torch.no_grad():
        ids = token_ids.unsqueeze(0).to(device)   # add batch dim
        model(ids)

    handle.remove()
    return hidden["states"][0]   # remove batch dim → [T, embed_dim]


class LinearProbe(nn.Module):
    """
    A logistic regression probe: linear layer + sigmoid (binary case).
    Trained on top of frozen hidden states.
    """

    def __init__(self, embed_dim: int, num_classes: int = 2):
        super().__init__()
        self.linear = nn.Linear(embed_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


def train_probe(
    hidden_states: torch.Tensor,   # [N, embed_dim]
    labels: torch.Tensor,          # [N] integer class labels
    num_classes: int = 2,
    epochs: int = 100,
    lr: float = 1e-2,
) -> float:
    """
    Train a linear probe and return its accuracy.
    Uses a simple train/test split (80/20).
    """
    N = len(labels)
    split = int(0.8 * N)
    X_train, X_test = hidden_states[:split], hidden_states[split:]
    y_train, y_test = labels[:split], labels[split:]

    probe = LinearProbe(hidden_states.shape[-1], num_classes)
    opt   = torch.optim.Adam(probe.parameters(), lr=lr)

    for _ in range(epochs):
        logits = probe(X_train)
        loss   = nn.functional.cross_entropy(logits, y_train)
        opt.zero_grad()
        loss.backward()
        opt.step()

    with torch.no_grad():
        preds = probe(X_test).argmax(dim=-1)
        acc   = (preds == y_test).float().mean().item()

    return acc


def probe_newline(model, token_ids: torch.Tensor, tokenizer, device: str = "cpu"):
    """
    Example probe: can we predict "is the next character a newline?"
    from the hidden state at each layer?

    A high accuracy means the model has encoded line-boundary information —
    it "knows" when a line is about to end.
    """
    newline_id = tokenizer.encode("\n")[0]
    # Labels: 1 if the next token is a newline, 0 otherwise
    labels = (token_ids[1:] == newline_id).long()

    print("=== Probing: newline prediction by layer ===\n")
    print(f"Base rate (always predict majority): {max(labels.float().mean().item(), 1 - labels.float().mean().item()):.3f}\n")

    for layer_idx in range(len(model.blocks)):
        # Hidden states at positions 0..T-1, predicting token at 1..T
        states = extract_hidden_states(model, token_ids[:-1], layer_idx, device)
        acc = train_probe(states, labels)
        bar = "█" * int(acc * 40)
        print(f"  Layer {layer_idx:2d}: {acc:.3f}  {bar}")


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    sample_train = (
        "Shall I compare thee to a summer's day?\n"
        "Thou art more lovely and more temperate.\n"
        "Rough winds do shake the darling buds of May,\n"
        "And summer's lease hath all too short a date.\n"
        "\n"
        "Sometime too hot the eye of heaven shines,\n"
        "And often is his gold complexion dimmed;\n"
    ) * 10

    sample_gen = (
        "The moonlight falls upon the evening sea,\n"
        "And shadows dance where gentle breezes blow.\n"
        "The silent stars look down on you and me,\n"
        "As waves below in silver currents flow.\n"
        "\n"
        "In dreams I walk the paths of long ago,\n"
        "Where roses bloomed and soft the songbirds sang.\n"
    ) * 10

    compare_corpora(sample_train, sample_gen)

    print("\n(Probing demo requires a trained model — see 06_training.py)")
