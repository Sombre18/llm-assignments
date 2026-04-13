"""
06_training.py
==============
Training the model: the optimisation loop.

The loop is conceptually simple:
  1. Sample a batch of (input, target) sequences.
  2. Forward pass: compute logits.
  3. Compute loss: how wrong were we?
  4. Backward pass: compute gradients (how should each weight change?).
  5. Optimiser step: nudge weights in the right direction.
  6. Repeat.

Everything interesting happens in the details.
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import math
from pathlib import Path


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class PoetryDataset(Dataset):
    """
    Wraps a tokenised corpus into (input, target) pairs for the DataLoader.

    PyTorch's Dataset protocol requires __len__ and __getitem__.
    DataLoader handles batching, shuffling, and parallel data loading.
    """

    def __init__(self, token_ids: list[int], context_length: int):
        self.data = torch.tensor(token_ids, dtype=torch.long)
        self.context_length = context_length

    def __len__(self) -> int:
        return len(self.data) - self.context_length

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        chunk = self.data[idx : idx + self.context_length + 1]
        return chunk[:-1], chunk[1:]   # input, target (shifted by 1)


# ---------------------------------------------------------------------------
# Learning rate schedule
# ---------------------------------------------------------------------------

def get_lr(step: int, warmup_steps: int, max_steps: int, max_lr: float, min_lr: float) -> float:
    """
    Linear warmup followed by cosine decay.

    Why warmup? At step 0, weights are random and gradients are noisy.
    A large learning rate would cause chaotic updates. We start small
    and ramp up over the first few hundred steps once the loss landscape
    becomes more predictable.

    Cosine decay then slowly reduces the learning rate as training
    converges, allowing fine-grained adjustments near the minimum.
    """
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    if step >= max_steps:
        return min_lr
    # Cosine decay between warmup and max_steps
    progress = (step - warmup_steps) / (max_steps - warmup_steps)
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * progress))


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(
    model: nn.Module,
    train_ids: list[int],
    val_ids: list[int],
    tokenizer,
    context_length: int = 256,
    batch_size: int = 32,
    max_steps: int = 20_000,
    max_lr: float = 3e-4,
    min_lr: float = 3e-5,
    warmup_steps: int = 500,
    eval_interval: int = 500,
    sample_interval: int = 2000,
    checkpoint_dir: str = "checkpoints",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    model = model.to(device)
    Path(checkpoint_dir).mkdir(exist_ok=True)

    train_dataset = PoetryDataset(train_ids, context_length)
    val_dataset   = PoetryDataset(val_ids,   context_length)

    # num_workers > 0 loads data in parallel with training — speeds things up.
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  num_workers=0, pin_memory=False)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False)
    

    # AdamW: Adam with decoupled weight decay.
    # Weight decay is a regulariser — it gently pushes weights toward zero,
    # preventing any single weight from becoming dominant.
    # We don't apply it to biases or layer norm parameters (common practice).
    decay_params     = [p for n, p in model.named_parameters() if p.dim() >= 2]
    no_decay_params  = [p for n, p in model.named_parameters() if p.dim() < 2]
    optimizer = torch.optim.AdamW([
        {"params": decay_params,    "weight_decay": 0.1},
        {"params": no_decay_params, "weight_decay": 0.0},
    ], lr=max_lr, betas=(0.9, 0.95))

    train_iter = iter(train_loader)
    step = 0

    print(f"Training on {device}. Steps: {max_steps}, batch: {batch_size}")
    print(f"Train tokens: {len(train_ids):,} | Val tokens: {len(val_ids):,}\n")

    while step < max_steps:
        # --- fetch batch (cycle through loader) ---
        try:
            x, y = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            x, y = next(train_iter)

        x, y = x.to(device), y.to(device)

        # --- learning rate update ---
        lr = get_lr(step, warmup_steps, max_steps, max_lr, min_lr)
        for group in optimizer.param_groups:
            group["lr"] = lr

        # --- forward + loss ---
        model.train()
        logits = model(x)                          # [B, T, vocab_size]
        B, T, V = logits.shape
        loss = nn.functional.cross_entropy(
            logits.view(B * T, V), y.view(B * T)
        )

        # --- backward ---
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping: if gradients are very large, scale them down.
        # Prevents a single bad batch from causing a catastrophic weight update.
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        step += 1

        # --- logging ---
        if step % 100 == 0:
            bpc = loss.item() / math.log(2)
            print(f"step {step:5d} | loss {loss.item():.4f} | bpc {bpc:.4f} | lr {lr:.2e}")

        # --- validation ---
        if step % eval_interval == 0:
            val_loss = evaluate(model, val_loader, device)
            val_bpc  = val_loss / math.log(2)
            print(f"\n  [val] step {step} | loss {val_loss:.4f} | bpc {val_bpc:.4f}\n")

        # --- generation sample ---
        if step % sample_interval == 0:
            sample = generate_sample(model, tokenizer, device, context_length)
            print(f"\n  [sample @ step {step}]\n{'-'*40}\n{sample}\n{'-'*40}\n")

        # --- checkpoint ---
        if step % 5000 == 0:
            path = Path(checkpoint_dir) / f"step_{step:06d}.pt"
            torch.save({"step": step, "model": model.state_dict(),
                        "optimizer": optimizer.state_dict()}, path)
            print(f"  Checkpoint saved: {path}")

    return model


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: str) -> float:
    model.eval()
    total_loss, n = 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        B, T, V = logits.shape
        loss = nn.functional.cross_entropy(logits.view(B * T, V), y.view(B * T))
        total_loss += loss.item()
        n += 1
        if n >= 50:   # cap validation batches for speed
            break
    return total_loss / n


@torch.no_grad()
def generate_sample(model, tokenizer, device, context_length, length=300, temperature=0.8):
    model.eval()
    prompt = torch.zeros(1, 1, dtype=torch.long, device=device)
    ids = model.generate(prompt, max_new_tokens=length,
                         temperature=temperature, top_k=40)
    return tokenizer.decode(ids[0].tolist())


# ---------------------------------------------------------------------------
# Demo (minimal smoke test — real training needs a corpus)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from language_modeling import CharTokenizer
    from model import CharGPT

    text = "To be or not to be, that is the question.\n" * 200
    tok  = CharTokenizer(text)
    ids  = tok.encode(text)
    split = int(0.9 * len(ids))

    model = CharGPT(vocab_size=tok.vocab_size, embed_dim=64, num_heads=4,
                    num_layers=2, context_length=64)

    print(f"Smoke-test training on tiny corpus ({len(ids)} tokens)...")
    train(model, ids[:split], ids[split:], tok,
          context_length=64, batch_size=16, max_steps=200,
          eval_interval=100, sample_interval=200, max_lr=3e-4)
