"""Level 3 — from-scratch WordPiece-style tokenizer (pure Python, no dependencies).

Vocabulary maps words -> integer ids, with subword-OOD fallback to characters,
so nothing ever maps to an error at inference time.
"""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

PAD, UNK, BOS, EOS = "<pad>", "<unk>", "<bos>", "<eos>"
SPECIALS = [PAD, UNK, BOS, EOS]

_WORD_RE = re.compile(r"[A-Za-z]+|[0-9]+|[^\sA-Za-z0-9]")


class Tokenizer:
    """Word-level tokenizer with byte-fallback for unknown words."""

    def __init__(self, vocab: dict[str, int] | None = None):
        self.vocab = vocab or {}
        self.id_to_token = {i: t for t, i in self.vocab.items()}

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def encode(self, text: str, add_specials: bool = True) -> list[int]:
        ids: list[int] = []
        if add_specials and BOS in self.vocab:
            ids.append(self.vocab[BOS])
        for word in _WORD_RE.findall(text.lower()):
            tid = self.vocab.get(word, self.vocab.get(UNK))
            if tid is not None:
                ids.append(tid)
            else:  # character fallback for unknown words
                ids.extend(
                    self.vocab.get(ch, self.vocab[UNK])
                    for ch in word
                    if ch in self.vocab or UNK in self.vocab
                )
        if add_specials and EOS in self.vocab:
            ids.append(self.vocab[EOS])
        return ids

    def decode(self, ids: list[int], skip_specials: bool = True) -> str:
        specials = set(SPECIALS) if skip_specials else set()
        parts = [self.id_to_token.get(i, "") for i in ids if self.id_to_token.get(i, "") not in specials]
        # rejoin: words get a space, punctuation attaches to previous token
        pieces: list[str] = []
        for tok in parts:
            if not tok:
                continue
            need_space = pieces and re.match(r"[A-Za-z0-9]", tok) and re.search(r"[A-Za-z0-9]$", pieces[-1])
            pieces.append((" " if need_space else "") + tok)
        return "".join(pieces).strip()

    @classmethod
    def train(cls, texts: list[str], vocab_size: int = 4096) -> "Tokenizer":
        counter: Counter[str] = Counter()
        for text in texts:
            counter.update(w.lower() for w in _WORD_RE.findall(text))
        vocab = {tok: i for i, tok in enumerate(SPECIALS)}
        for word, _freq in counter.most_common(vocab_size - len(SPECIALS)):
            if word not in vocab:
                vocab[word] = len(vocab)
        # make sure every ASCII char is in vocab for fallback
        for ch in "abcdefghijklmnopqrstuvwxyz0123456789.,!?;:'\"-()[] \n":
            if ch not in vocab:
                vocab[ch] = len(vocab)
        return cls(vocab)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.vocab, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "Tokenizer":
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))
