"""Level 3 — dataset preparation.

Builds a corpus from .txt files in a data folder and caches token ids to a
binary file so repeated training runs skip re-tokenization.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from .tokenizer import Tokenizer


def collect_texts(data_dir: Path) -> list[str]:
    texts: list[str] = []
    for path in sorted(data_dir.rglob("*.txt")):
        content = path.read_text(encoding="utf-8", errors="ignore")
        if content.strip():
            texts.append(content)
    if not texts:
        raise FileNotFoundError(f"No .txt files found under {data_dir}. Add some first.")
    return texts


def prepare(data_dir: str | Path = "data", vocab_size: int = 4096, cache_dir: str | Path = "askwise_model/cache") -> Tokenizer:
    data_dir = Path(data_dir)
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    texts = collect_texts(data_dir)
    tokenizer = Tokenizer.train(texts, vocab_size=vocab_size)
    tokenizer.save(cache_dir / "vocab.json")

    ids: list[int] = []
    for text in texts:
        ids.extend(tokenizer.encode(text, add_specials=False))
    ids_path = cache_dir / "train_ids.bin"
    # store as uint16 if vocab fits, else uint32
    import struct

    code = "H" if tokenizer.vocab_size <= 0xFFFF else "I"
    with open(ids_path, "wb") as f:
        f.write(struct.pack(f"<{len(ids)}{code}", *ids))
    (cache_dir / "dtype.txt").write_text(code, encoding="ascii")
    print(f"prepared {len(texts)} docs, {len(ids):,} tokens, vocab={tokenizer.vocab_size}")
    print(f"cached ids to {ids_path}")
    return tokenizer


def load_train_ids(cache_dir: str | Path = "askwise_model/cache") -> tuple[list[int], str]:
    cache_dir = Path(cache_dir)
    code = (cache_dir / "dtype.txt").read_text(encoding="ascii").strip()
    import struct

    data = (cache_dir / "train_ids.bin").read_bytes()
    n = len(data) // (2 if code == "H" else 4)
    ids = list(struct.unpack(f"<{n}{code}", data))
    return ids, code


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare Askwise training data.")
    parser.add_argument("--data", default="data", help="folder containing .txt files")
    parser.add_argument("--vocab", type=int, default=4096)
    args = parser.parse_args()
    prepare(data_dir=args.data, vocab_size=args.vocab)
