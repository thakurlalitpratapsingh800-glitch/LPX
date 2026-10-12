"""Level 3 — training loop for MiniGPT with periodic checkpoints.

Usage (from project root):
    python -m askwise_model.train --data ./data --steps 2000
    python -m askwise_model.train --resume askwise_model/checkpoints/last.pt

Checkpoints are saved to askwise_model/checkpoints/ as last.pt (resume point)
and best.pt (lowest validation loss).
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from .dataset import load_train_ids, prepare
from .tokenizer import Tokenizer
from .transformer import MiniGPT, count_parameters

CHECKPOINT_DIR = Path(__file__).resolve().parent / "checkpoints"


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"  # Apple silicon (not typical on Windows but harmless)
    return "cpu"


def batches(ids: list[int], batch_size: int, block_size: int, device: str):
    """Yield (x, y) where x=inputs and y=targets shifted by one."""
    span = len(ids) - block_size - 1
    if span <= 0:
        return
    n_possible = max(1, span // max(1, batch_size // 4))
    for _ in range(n_possible):
        starts = torch.randint(0, span, (batch_size,))
        x = torch.stack([torch.tensor(ids[s : s + block_size]) for s in starts])
        y = torch.stack([torch.tensor(ids[s + 1 : s + 1 + block_size]) for s in starts])
        yield x.to(device), y.to(device)


def loss_fn(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits.view(-1, logits.size(-1)), targets.reshape(-1))


@torch.no_grad()
def evaluate(model: MiniGPT, ids: list[int], device: str, batch_size: int = 32) -> float:
    model.eval()
    losses = []
    for x, y in batches(ids, batch_size, 128, device):
        logits = model(x)
        losses.append(loss_fn(logits, y).item())
        if len(losses) >= 8:  # cap eval cost on small datasets
            break
    model.train()
    return sum(losses) / len(losses)


def save_checkpoint(model, optimizer, step, best_val, config, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": step,
            "best_val": best_val,
            "config": config,
        },
        path,
    )


def train(args) -> None:
    device = get_device()
    print(f"training on device: {device}")

    # make sure the tokenizer/data are prepared once, then load cached ids
    cache_dir = Path(args.cache)
    if not (cache_dir / "train_ids.bin").exists():
        prepare(data_dir=args.data, vocab_size=args.vocab, cache_dir=cache_dir)

    ids, _code = load_train_ids(cache_dir)
    tokenizer = Tokenizer.load(cache_dir / "vocab.json")

    # split off a small validation slice (train first, tail held out for val)
    val_n = max(1, int(len(ids) * 0.05))
    train_ids, val_ids = ids[:-val_n], ids[-val_n:]

    model = MiniGPT(
        vocab_size=tokenizer.vocab_size,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        max_seq_len=args.block_size,
        dropout=args.dropout,
    )
    print(f"parameters: {count_parameters(model):,}")
    model.to(device)

    checkpoint_config = {
        "d_model": args.d_model,
        "n_heads": args.n_heads,
        "n_layers": args.n_layers,
        "max_seq_len": args.block_size,
        "vocab_size": tokenizer.vocab_size,
    }

    def save(step: int, val_loss: float, path: Path, best_val: float) -> None:
        save_checkpoint(model, optimizer, step, best_val, checkpoint_config, path)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.1)

    ckpt = None
    if args.resume and Path(args.resume).exists():
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        print(f"resumed from {args.resume} at step {ckpt.get('step', 0)}")

    start_step = ckpt.get("step", 0) if ckpt else 0
    if not isinstance(start_step, int):
        start_step = 0
    best_val = ckpt.get("best_val", float("inf")) if ckpt else float("inf")

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr, total_steps=max(1, args.steps - start_step), pct_start=0.1
    )

    model.train()
    step = start_step
    t0 = time.time()
    while step < args.steps:
        epoch_batches = list(batches(train_ids, args.batch_size, args.block_size, device))
        if not epoch_batches:
            raise RuntimeError(
                "Not enough token data to build a batch. Add more text to ./data "
                "(or reduce --block-size / --batch-size)."
            )
        for x, y in epoch_batches:
            if step >= args.steps:
                break
            logits = model(x)
            loss = loss_fn(logits, y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            step += 1
            if step % args.log_interval == 0:
                lr_now = scheduler.get_last_lr()[0]
                print(
                    f"step {step}/{args.steps}  loss {loss.item():.3f}  "
                    f"lr {lr_now:.2e}  {time.time() - t0:.0f}s"
                )
            if step % args.eval_interval == 0 or step == args.steps:
                val_loss = evaluate(model, val_ids, device, args.batch_size)
                print(f"  val loss {val_loss:.3f}  (best {best_val:.3f})")
                save(step, best_val, CHECKPOINT_DIR / "last.pt", best_val)
                if val_loss < best_val:
                    best_val = val_loss
                    save(step, best_val, CHECKPOINT_DIR / "best.pt", best_val)
                    print("  saved new best checkpoint")
                model.train()

    save(step, best_val, CHECKPOINT_DIR / "last.pt", best_val)
    print(f"training done in {time.time() - t0:.0f}s. checkpoints in {CHECKPOINT_DIR}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train the Askwise MiniGPT model.")
    p.add_argument("--data", default="data", help="folder of .txt files")
    p.add_argument("--cache", default="askwise_model/cache")
    p.add_argument("--vocab", type=int, default=4096)
    p.add_argument("--d-model", type=int, default=256)
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--n-layers", type=int, default=4)
    p.add_argument("--block-size", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--log-interval", type=int, default=50)
    p.add_argument("--eval-interval", type=int, default=200)
    p.add_argument("--resume", type=str, default="")
    return p


if __name__ == "__main__":
    train(build_parser().parse_args())
