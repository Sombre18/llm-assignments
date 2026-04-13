"""
01_language_modeling.py
=======================
What is a language model, really?

A language model assigns a probability to sequences of tokens.
In practice, we train it to do one thing: predict the next token
given everything that came before.

That single objective — next-token prediction — is surprisingly powerful.
To predict well, the model must implicitly learn grammar, style,
facts, and structure. It has no other choice.
"""

import torch
import torch.nn.functional as F
from collections import Counter


# ---------------------------------------------------------------------------
# The core task, stated plainly
# ---------------------------------------------------------------------------

def next_token_probability(logits: torch.Tensor) -> torch.Tensor:
    """
    A model's output is raw scores (logits) over the vocabulary.
    We turn those into probabilities with softmax.

    logits: shape [vocab_size]  — one score per possible next token
    returns: shape [vocab_size] — a probability distribution
    """
    return F.softmax(logits, dim=-1)


# ---------------------------------------------------------------------------
# Character-level tokenization
# ---------------------------------------------------------------------------

class CharTokenizer:
    """
    The simplest possible tokenizer: each character is one token.

    Vocabulary = the set of distinct characters in the corpus.
    Every character gets an integer ID; every integer maps back to a character.

    Tradeoffs vs. subword tokenization (BPE, SentencePiece):
    - Pro: tiny vocabulary, no segmentation ambiguity, captures phonology
      naturally (useful for rhyme and meter).
    - Con: sequences are much longer; the model must learn that
      ['c','h','a','t'] form a unit before reasoning about meaning.
      Subword models get that compression for free.

    For poetry, the tradeoff is defensible. A character-level model
    can learn line length, syllable patterns, and rhyme directly.
    """

    def __init__(self, text: str):
        chars = sorted(set(text))
        self.vocab = chars
        self.vocab_size = len(chars)
        self._ch2id = {ch: i for i, ch in enumerate(chars)}
        self._id2ch = {i: ch for i, ch in enumerate(chars)}

    def encode(self, text: str) -> list[int]:
        return [self._ch2id[ch] for ch in text]

    def decode(self, ids: list[int]) -> str:
        return "".join(self._id2ch[i] for i in ids)


# ---------------------------------------------------------------------------
# Building training examples from raw text
# ---------------------------------------------------------------------------

def make_sequences(token_ids: list[int], context_length: int) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Slice a long token sequence into (input, target) pairs.

    For a context_length of 4 and the text "hello":
        input:  [h, e, l, l]
        target: [e, l, l, o]

    Every position in `input` has a corresponding next token in `target`.
    During training, the model sees input[0..t] and must predict target[t]
    for every t simultaneously. This is what makes Transformer training
    efficient — all positions train in parallel.

    context_length: how many characters the model sees at once.
    Choose it large enough to span a few lines of poetry (~256 is reasonable),
    but larger means more memory per batch.
    """
    data = torch.tensor(token_ids, dtype=torch.long)
    inputs, targets = [], []

    for start in range(len(data) - context_length):
        inputs.append(data[start : start + context_length])
        targets.append(data[start + 1 : start + context_length + 1])

    return torch.stack(inputs), torch.stack(targets)


# ---------------------------------------------------------------------------
# The training objective
# ---------------------------------------------------------------------------

def language_modeling_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Cross-entropy loss: how surprised is the model by the actual next token?

    logits:  [batch, sequence_length, vocab_size]
    targets: [batch, sequence_length]

    We flatten batch and sequence into one dimension because PyTorch's
    cross_entropy expects [N, vocab_size] and [N].

    Lower loss = the model assigns higher probability to the correct token.
    Perfect prediction = loss of 0. Random guessing over V characters
    gives loss = log(V). That's your baseline to beat.
    """
    B, T, V = logits.shape
    return F.cross_entropy(logits.view(B * T, V), targets.view(B * T))


def bits_per_character(loss: float) -> float:
    """
    Convert nats (natural log loss) to bits (log base 2).
    More interpretable: how many bits does the model use per character?
    A good model on English text reaches ~1.3–1.5 bits/char.
    Uniform random over 26 letters = ~4.7 bits/char.
    """
    import math
    return loss / math.log(2)


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    sample_text = (
        "Shall I compare thee to a summer's day?\n"
        "Thou art more lovely and more temperate.\n"
    )

    tok = CharTokenizer(sample_text)
    print(f"Vocabulary size: {tok.vocab_size}")
    print(f"Vocabulary: {tok.vocab}")

    ids = tok.encode(sample_text)
    print(f"\nEncoded (first 20 tokens): {ids[:20]}")
    print(f"Decoded back: '{tok.decode(ids[:20])}'")

    inputs, targets = make_sequences(ids, context_length=8)
    print(f"\nDataset shape — inputs: {inputs.shape}, targets: {targets.shape}")
    print(f"First input:  {inputs[0].tolist()} → '{tok.decode(inputs[0].tolist())}'")
    print(f"First target: {targets[0].tolist()} → '{tok.decode(targets[0].tolist())}'")

    # Simulate what a random model's loss looks like
    random_logits = torch.randn(1, 8, tok.vocab_size)
    loss = language_modeling_loss(random_logits, targets[:1])
    print(f"\nRandom model loss: {loss.item():.3f} nats")
    print(f"Random model loss: {bits_per_character(loss.item()):.3f} bits/char")
    print(f"Theoretical random baseline: {bits_per_character(__import__('math').log(tok.vocab_size)):.3f} bits/char")
