# LPX AI

A free, open-source AI platform with three levels, built on Python, Flask, SQLite, Ollama and PyTorch. **No paid APIs, no API keys, no subscriptions.**

---

## What's inside

| Level | Feature | Where |
|---|---|---|
| 1 | **AI search engine + chatbot:** Direct answers with citations, conversation memory (SQLite), PDF uploads, free web search with automatic fallback | `app.py`, `askwise_api/` |
| 2 | **Your own AI server:** Local, open-weight models via Ollama — fully offline inference, no API keys required | `askwise_api/ollama_client.py` |
| 3 | **Your own AI model:** A Transformer language model written from scratch in PyTorch — tokenizer, dataset preparation, training, checkpoints and generation | `askwise_model/` |

The original green color palette, Manrope/DM Sans fonts, hero search and source cards are preserved under the new LPX branding. The hero search is now the entry point to a chat.

---

## Quick start (Windows)

```bat
:: 0. Prerequisites: Python 3.10+ from python.org.
:: Optional: Install Ollama for local AI.
winget install --id Ollama.Ollama --exact --source winget

:: Download a model (choose one that fits your system).
ollama pull qwen3:0.6b

:: 1. One-click setup.
setup_windows.bat

:: 2. Run the platform.
.venv\Scripts\activate
python app.py

:: 3. Open http://127.0.0.1:5000 in your browser.
```

### Manual setup (any OS)

```bash
python -m venv .venv
```

On Windows:

```bat
.venv\Scripts\activate
```

On Linux/macOS:

```bash
source .venv/bin/activate
```

Install dependencies and start the app:

```bash
pip install -r requirements.txt
python app.py
```

---

## Using each level

### Level 1 — AI search engine + chatbot

- **Direct answers:** Ask questions through the main search box and continue the conversation in the chat interface.
- **Conversation memory:** The last 8 conversation turns are stored in SQLite (`askwise.db`).
- **Citations:** Web-grounded answers can include numbered sources with clickable links.
- **PDF uploads:** Use the attachment button to upload PDFs. Extracted text can be used to answer questions.
- **Free search chain:** DuckDuckGo (`ddgs`) → Wikipedia API fallback.
- **Fallback answers:** If web search is unavailable, the app can use the local model or uploaded PDF text when available.
- **Controls:** Use the available toggles to disable web search or local-model responses per message.

### Level 2 — Your own AI server (Ollama)

Install Ollama, then download a model:

```bash
ollama pull qwen3:0.6b
```

Ollama usually starts automatically after installation. If necessary, start its service using:

```bash
ollama serve
```

- The app detects available Ollama models automatically.
- The health endpoint (`/api/health`) can report the local AI status.
- Downloaded models run locally, without requiring an external AI API key.
- Once the model is downloaded, inference can work offline, provided all required local services and dependencies are available.

**Hardware guidance:** Actual requirements depend on the model, quantization, context length and other applications running on your computer. Smaller models such as `qwen3:0.6b` are a good starting point for lower-resource systems.

### Level 3 — Your own Transformer model (PyTorch, from scratch)

Put your own `.txt` files into the `data` directory. A sample corpus may already be included in the project.

Build the dataset and tokenizer:

```bash
python -m askwise_model.dataset --data data
```

Train the model:

```bash
python -m askwise_model.train --data data --steps 5000
```

Resume training from a checkpoint:

```bash
python -m askwise_model.train --resume askwise_model/checkpoints/last.pt
```

Generate text:

```bash
python -m askwise_model.generate --prompt "artificial intelligence" --tokens 100
```

Use interactive mode:

```bash
python -m askwise_model.generate --interactive
```

The model implementation in `askwise_model/transformer.py` includes components such as causal multi-head self-attention and feed-forward blocks.

- **Tokenizer:** Converts text into tokens for training and generation.
- **Training:** Learns patterns from the supplied dataset.
- **Checkpoints:** Save model progress so training can be resumed.
- **Text generation:** Generates text from a supplied prompt.
- **No pretrained model required:** This level is designed to train its own small model rather than relying on a downloaded pretrained language model.

The default model configuration is intended for experimentation and learning. Training speed and memory usage depend on your computer, dataset and configuration. A small model trained on a small corpus will not have the knowledge or capabilities of ChatGPT.

---

## Project layout

```text
app.py
    Flask application: chat, search, uploads and health APIs

askwise_api/
    search_engine.py
        Free search engines, fallbacks and answer extraction
    database.py
        SQLite schema for conversations, messages and documents
    rag.py
        Answer pipeline: memory, PDFs, web search and model routing
    ollama_client.py
        Level 2 local inference and model detection
    pdf_reader.py
        PDF text extraction

askwise_model/
    tokenizer.py
        From-scratch tokenizer
    transformer.py
        Transformer model implementation
    dataset.py
        Dataset preparation and token caching
    train.py
        Training loop, checkpoints and resume support
    generate.py
        Text-generation CLI and generate_text() API

templates/
    index.html
        Main website interface

static/
    style.css
        Website styling
    script.js
        Chat interface and frontend logic

data/
    sample_corpus.txt
        Starter dataset for the custom model

setup_windows.bat
    Windows setup script

askwise.db
    SQLite database created at runtime
```

**Note:** The internal `askwise_api/` and `askwise_model/` names are intentionally unchanged. Renaming them would require updating Python imports and other references throughout the project.

---

## Known limitations of the free stack

- **Search quality:** DuckDuckGo's community endpoint (`ddgs`) may occasionally be rate-limited or change its response format. Wikipedia and local fallbacks may provide less current information.
- **Model capability:** Small local models can make mistakes, misunderstand questions or generate incorrect information. Verify important answers.
- **Your own Transformer:** A small model demonstrates how language-model training works. It will not provide encyclopedic answers when trained on a tiny sample dataset.
- **Training requirements:** Larger datasets, longer training and bigger models require more computing resources.
- **PDF OCR:** Scanned or image-only PDFs need OCR before their text can be extracted by a text-based PDF reader.
- **Offline operation:** Local inference can work without internet after installation and model downloads. Web search and other online features still require an internet connection.

---

## Project philosophy

LPX AI is designed to help you understand and build AI systems step by step:

1. **Search and chat** — build a useful AI-powered application.
2. **Local AI server** — run an open-weight language model on your own computer.
3. **Custom AI model** — learn how tokenization, Transformers, training and text generation work.

**LPX AI — Your Intelligence. Your AI.**

An open-source learning project focused on understanding and building AI with Python.