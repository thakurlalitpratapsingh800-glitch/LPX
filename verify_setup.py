"""Askwise self-check: run `python verify_setup.py` after installing requirements.

Verifies, without needing the web:
  1. every module imports
  2. the tokenizer round-trips text
  3. the from-scratch Transformer runs a forward + generate pass
  4. the SQLite layer stores and recalls a conversation
  5. Ollama reachability (informational, not required to pass)
"""
from __future__ import annotations

import tempfile
from pathlib import Path

ok = True


def check(name: str, fn) -> None:
    global ok
    try:
        fn()
        print(f"  PASS  {name}")
    except Exception as exc:  # noqa: BLE001 - report everything
        ok = False
        print(f"  FAIL  {name}: {exc}")


def test_imports() -> None:
    import app  # noqa: F401
    import askwise_api.database
    import askwise_api.pdf_reader
    import askwise_api.rag
    import askwise_api.search_engine
    import askwise_model.dataset
    import askwise_model.generate
    import askwise_model.tokenizer
    import askwise_model.transformer
    import askwise_model.train


def test_tokenizer() -> None:
    from askwise_model.tokenizer import Tokenizer

    tok = Tokenizer.train(["hello wonderful world", "the world of transformers"], vocab_size=64)
    ids = tok.encode("Hello, wonderful WORLD!")
    text = tok.decode(ids)
    assert "hello" in text and "world" in text, text
    unknown = tok.encode("xylophonezzz")
    assert tok.decode(unknown), "character fallback failed"


def test_transformer_forward_and_generate() -> None:
    import torch

    from askwise_model.transformer import MiniGPT

    model = MiniGPT(vocab_size=64, d_model=32, n_heads=4, n_layers=2, max_seq_len=32)
    x = torch.randint(0, 64, (2, 16))
    logits = model(x)
    assert logits.shape == (2, 16, 64), logits.shape
    out = model.generate(x[:1], max_new_tokens=5, temperature=1.0, top_k=10)
    assert out.shape[1] == 21, out.shape


def test_training_one_step(tmp: Path) -> None:
    import argparse

    from askwise_model.train import train

    data = tmp / "data"
    data.mkdir()
    (data / "t.txt").write_text(
        "hello world this is a test corpus for the tiny askwise transformer. " * 20,
        encoding="utf-8",
    )
    args = argparse.Namespace(
        data=str(data), cache=str(tmp / "cache"), vocab=128, d_model=16, n_heads=2,
        n_layers=1, block_size=16, batch_size=4, steps=2, lr=1e-3, dropout=0.0,
        log_interval=1, eval_interval=2, resume="",
    )
    train(args)
    from askwise_model.train import CHECKPOINT_DIR

    assert (CHECKPOINT_DIR / "last.pt").exists(), "checkpoint was not saved"


def test_database(tmp: Path) -> None:
    import askwise_api.database as db

    db.DB_PATH = tmp / "test.db"
    db._thread_local.conn = None  # force reconnect to test db
    db.init_db()
    cid = db.create_conversation("test")
    db.add_message(cid, "user", "What is AI?")
    db.add_message(cid, "assistant", "AI is intelligence in machines.", mode="local")
    history = db.memory_history(cid)
    assert history == [("What is AI?", "AI is intelligence in machines.")], history


def test_ollama_status() -> None:
    from askwise_api.ollama_client import best_model_for_ram, model_available

    model = model_available()
    if model:
        print(f"          (Ollama online, using {model})")
    else:
        print(f"          (Ollama offline -> web/PDF answers still work. Recommended: {best_model_for_ram()})")


if __name__ == "__main__":
    print("Askwise self-check")
    check("imports", test_imports)
    check("tokenizer roundtrip", test_tokenizer)
    check("transformer forward + generate", test_transformer_forward_and_generate)
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        check("tiny training run (2 steps)", lambda: test_training_one_step(tmp))
        check("sqlite memory", lambda: test_database(tmp))
    check("ollama status", test_ollama_status)
    print("ALL PASS" if ok else "SOME CHECKS FAILED - read the lines above")
    raise SystemExit(0 if ok else 1)
