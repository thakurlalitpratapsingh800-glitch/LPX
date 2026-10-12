"""Level 3 — generate text with your own trained Transformer model.

CLI usage (from project root):
    python -m askwise_model.generate --prompt "Once upon a time" --tokens 100
    python -m askwise_model.generate --interactive

Also exposes `generate_text()` so the Flask app can call your own model.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from .tokenizer import Tokenizer
from .transformer import MiniGPT

CHECKPOINT_DIR = Path(__file__).resolve().parent / "checkpoints"
CACHE_DIR = Path(__file__).resolve().parent / "cache"


def _load_model(device: str) -> tuple[MiniGPT, Tokenizer]:
    if not (CACHE_DIR / "vocab.json").exists():
        raise FileNotFoundError("No tokenizer found. Run `python -m askwise_model.dataset` first.")
    tokenizer = Tokenizer.load(CACHE_DIR / "vocab.json")

    ckpt_path = CHECKPOINT_DIR / "best.pt"
    if not ckpt_path.exists():
        ckpt_path = CHECKPOINT_DIR / "last.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(
            "No checkpoint found. Train first: `python -m askwise_model.train`."
        )
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    config = ckpt.get("config", {})
    model = MiniGPT(
        vocab_size=tokenizer.vocab_size,
        d_model=config.get("d_model", 256),
        n_heads=config.get("n_heads", 4),
        n_layers=config.get("n_layers", 4),
        max_seq_len=config.get("max_seq_len", 256),
    )
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()
    return model, tokenizer


def generate_text(
    prompt: str,
    max_new_tokens: int = 80,
    temperature: float = 0.8,
    top_k: int = 50,
    device: str | None = None,
) -> str:
    """Generate a continuation for `prompt` using your own MiniGPT checkpoint."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer = _load_model(device)
    ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
    out = model.generate(
        ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        eos_id=tokenizer.vocab.get("<eos>"),
    )
    return tokenizer.decode(out[0].tolist())


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate text with the Askwise MiniGPT.")
    parser.add_argument("--prompt", default="")
    parser.add_argument("--tokens", type=int, default=80)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--interactive", action="store_true")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")
    model, tokenizer = _load_model(device)

    if args.interactive:
        print("Type a prompt (empty line quits).")
        while True:
            try:
                prompt = input("prompt> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not prompt:
                break
            print(generate_text(prompt, args.tokens, args.temperature, args.top_k))
    else:
        print(generate_text(args.prompt, args.tokens, args.temperature, args.top_k))


if __name__ == "__main__":
    main()
