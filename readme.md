# Crisis RAG

An **offline-first, intent-aware crisis assistance system**. It answers disaster and first-aid
questions entirely on a local device — no internet connection required at query time — by
combining a curated crisis knowledge base, a local small language model (Phi-3 Mini), and an
agentic reasoning layer that classifies the *type* of emergency before deciding how to answer it.

When the device does happen to have internet access, it can also opportunistically check for
updated versions of its source documents in the background, without interrupting normal use.

---

## Why this exists

In an actual disaster, internet connectivity is often the first thing to fail. A crisis assistant
that depends on a live connection to an LLM API is useless exactly when it's needed most. Crisis
RAG is built the other way around: everything it needs to answer a question — the knowledge base,
the embedding model, and the language model — lives on disk. Connectivity is treated as a bonus,
not a requirement.

The second idea behind this project is that **not all crisis questions should be answered the same
way**. "My leg is bleeding and won't stop" and "what's the safest evacuation route for a flood"
need different retrieval strategies, different tones, and different urgency thresholds. Most RAG
chatbots treat every question identically. This system classifies intent *before* retrieving
anything, and routes accordingly.

---

## Features

- **Fully offline question-answering** — retrieval, embedding, and generation all run locally.
- **Intent-aware triage** — every query is classified into one of five categories before retrieval:
  `MEDICAL`, `EVACUATION`, `SURVIVAL`, `EMOTIONAL`, `GENERAL`. Each category gets its own
  retrieval strictness, system prompt, and response style. A hybrid keyword + LLM classifier makes
  the call, with keyword-based medical signals always taking safety-first priority over the model.
- **ReAct-style agent loop** — if the first retrieval attempt doesn't return anything relevant, the
  agent reformulates the query and tries again (max 2 iterations) before falling back to an honest
  "I don't have that information, call emergency services" response.
- **Typo-robust retrieval** — a lightweight query normalizer catches likely typos in hazard-specific
  terms (e.g. "lamdlside" → also considers "landslide") before embedding or keyword matching, so a
  single misspelled word doesn't silently break retrieval.
- **Source-cited answers** — every answer reports which knowledge base document(s) it drew from,
  and their similarity scores.
- **Background sync** — when online, a background thread can check configured source URLs for
  updated documents (via checksum comparison) and re-ingest only what changed, without touching
  anything else in the knowledge base.
- **Evaluation harness** — a 24-case test suite compares three configurations (plain RAG, RAG +
  agent without triage, and the full system) on retrieval quality, intent accuracy, refusal
  correctness, and latency.

---

## How it works — the pipeline

### 1. Ingestion (build the knowledge base)
```
PDF / .txt / .md files in knowledge_base/
        │
        ▼
  extract text  (PyMuPDF for PDFs)
        │
        ▼
  split into ~500-token chunks
        │
        ▼
  embed each chunk  (sentence-transformers, all-MiniLM-L6-v2, 384-dim)
        │
        ▼
  store in LanceDB  (data/lancedb/) — a local, file-based vector database
```

### 2. Answering a question (full agent)
```
user query
        │
        ▼
  normalize query        (fix likely typos in hazard-specific terms)
        │
        ▼
  classify intent        (keyword rules + local LLM, medical signals always win)
        │
        ▼
  retrieve top-3 chunks   (intent-specific similarity threshold)
        │
        ├─ good match ──────────────► generate answer (intent-specific prompt) ─► return
        │
        └─ weak/no match ─► reformulate query ─► retrieve again ─► generate (or honest fallback)
```

### 3. Background sync (opportunistic, only when online)
```
every 30 min:
  online?
    │
    ├─ no  ──► skip, try again next interval
    │
    └─ yes ──► for each configured source:
                 download it
                 compare SHA-256 checksum to last known version
                 changed? ──► replace file in knowledge_base/, delete old chunks,
                               re-chunk + re-embed + re-add, update manifest
                 unchanged? ─► do nothing
```

---

## Project structure

```
crisis_rag/
├── knowledge_base/          curated crisis/first-aid PDFs, .txt, .md source documents
├── data/lancedb/            the vector database (built by ingestion, not committed to git)
├── shared/
│   └── query_normalizer.py  typo correction for hazard-specific terms
├── phase_one/                # knowledge base construction
│   ├── chunker.py
│   ├── embedder.py
│   ├── vector_store.py
│   └── ingest.py
├── phase_two/                 # offline RAG pipeline
│   ├── llm.py                 Phi-3 Mini wrapper (llama.cpp)
│   ├── rag_pipeline.py        retrieve → generate, single-pass, no agent
│   └── query.py               CLI for the plain RAG pipeline
├── phase_three/                # agentic core
│   ├── triage_classifier.py    5-way intent classification
│   ├── agent.py                ReAct loop + intent-specific retrieval/prompting
│   └── chat.py                 CLI for the full agent (this is the main entry point)
├── sync/                        # opportunistic background sync
│   ├── sync_manager.py
│   ├── sources.json             configure which documents to auto-check for updates
│   └── manifest.json            tracks checksums (created automatically)
└── phase_four/evaluation/       # comparison harness for the paper
    ├── test_cases.json
    ├── configs.py
    ├── metrics.py
    └── run_eval.py
```

---

## Requirements

- **Python 3.10 or newer**
- **~4 GB free disk space** — the Phi-3 Mini model file is ~2.4 GB, the embedding model is
  ~90 MB (downloaded automatically on first run), plus the knowledge base and vector database.
- **~8 GB RAM recommended** to run the language model comfortably on CPU.
- A **C++ build toolchain**, needed by `llama-cpp-python` to compile llama.cpp on install:
  - **Windows:** [Visual Studio Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) ("Desktop development with C++" workload)
  - **macOS:** Xcode Command Line Tools — `xcode-select --install`
  - **Linux:** `build-essential` (Debian/Ubuntu) or equivalent (`gcc`, `make`)
- No GPU required — `llama-cpp-python` runs Phi-3 Mini on CPU at a usable speed for this use case.
  An NVIDIA GPU with CUDA is optional if you want faster inference (see `llama-cpp-python`'s docs
  for `CMAKE_ARGS` build flags).

---

## Setup

### 1. Get the code

```
git clone https://github.com/gunnseth55/Crisis_rag.git
cd Crisis_rag
```

### 2. Create and activate a virtual environment

**Windows — Command Prompt (cmd.exe):**
```
python -m venv venv
venv\Scripts\activate.bat
```

**Windows — PowerShell:**
```
python -m venv venv
venv\Scripts\Activate.ps1
```
> If PowerShell blocks the script with an execution-policy error, run this once first:
> `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`

**macOS / Linux (bash or zsh):**
```
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```
pip install -r requirements.txt
```

This installs `sentence-transformers`, `lancedb`, `pyarrow`, `tqdm`, `pymupdf`, and
`llama-cpp-python`.

> **Note on `llama-cpp-python`:** this package compiles a native extension. Prebuilt wheels are
> available for common platforms and Python versions, so `pip install` usually just works. If it
> tries to build from source and fails, install the C++ toolchain listed under Requirements above
> first, then retry. If you have an NVIDIA GPU and want it used instead of CPU:
> ```
> CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python
> ```
> (macOS/Linux syntax; on Windows PowerShell use
> `$env:CMAKE_ARGS="-DGGML_CUDA=on"; pip install llama-cpp-python`)

### 4. Download the language model

Crisis RAG uses **Phi-3 Mini 4K Instruct**, quantized to GGUF format, from Microsoft's official
Hugging Face repository.

Download `Phi-3-mini-4k-instruct-q4.gguf` (~2.4 GB) from:
`https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf`

Either download it directly from that page in a browser, or via the command line
(requires `pip install huggingface_hub` first):
```
huggingface-cli download microsoft/Phi-3-mini-4k-instruct-gguf Phi-3-mini-4k-instruct-q4.gguf --local-dir . --local-dir-use-symlinks False
```

Save it anywhere on disk — you'll pass its path with `--model` in every command below.

### 5. Add your source documents

Place your crisis/first-aid PDFs, `.txt`, or `.md` files into `knowledge_base/`. A curated set
(NDMA, Red Cross, WHO, CDC guidance) is already included in this repo.

### 6. Build the knowledge base (run ingestion)

```
python phase_one/ingest.py
```

This reads every supported file in `knowledge_base/`, chunks it, embeds it, and stores it in
`data/lancedb/`. Already-ingested sources are skipped automatically, so it's safe to re-run after
adding new documents.

Optional — sanity-check retrieval quality with 5 canned test queries:
```
python phase_one/ingest.py --test
```

> To force a document to be re-ingested after editing it, delete its existing chunks first:
> ```
> python -c "from phase_one.vector_store import VectorStore; s = VectorStore('data/lancedb'); s.init(); s.delete_source('SOURCE_NAME')"
> ```

---

## Running the assistant

### Option A — Full agent (recommended, main entry point)

Intent-aware triage, ReAct reformulation loop, and background sync all included.

**Windows:**
```
python phase_three\chat.py --model C:\path\to\Phi-3-mini-4k-instruct-q4.gguf
```

**macOS / Linux:**
```
python phase_three/chat.py --model /path/to/Phi-3-mini-4k-instruct-q4.gguf
```

In-chat commands: `history`, `clear`, `sync` (manually trigger a sync check), `quit`.

### Option B — Plain RAG pipeline (no agent, no triage)

Useful for comparison or a simpler/faster baseline.

**Windows:**
```
python phase_two\query.py --model C:\path\to\Phi-3-mini-4k-instruct-q4.gguf
```

**macOS / Linux:**
```
python phase_two/query.py --model /path/to/Phi-3-mini-4k-instruct-q4.gguf
```

Optional flag for either option: `--db <path>` (defaults to `data/lancedb`).

---

## Background sync (keeping the knowledge base current)

Sync runs automatically inside `phase_three/chat.py`. To configure or test it directly:

1. Edit `sync/sources.json` and add real, verified download URLs for documents you want
   auto-checked (see the file's `_readme` field for the exact format). It ships empty — a wrong
   URL would silently replace a correct document with the wrong content, so nothing is pre-filled.

2. Run a one-off check (useful for testing):
```
   python sync/sync_manager.py --once
```

3. Or run it continuously in the background (Ctrl+C to stop):
```
   python sync/sync_manager.py --daemon
```
   Optional: `--interval <seconds>` to change the poll interval (default 1800 = 30 min).

Some official sources (e.g. CDC, WHO's IRIS repository) block automated downloads outright this
is a known limitation, not a bug in the sync mechanism itself.

---

## Evaluation (Phase 4 comparison harness)

Runs a 24-case test suite through three configurations — plain RAG, agent without triage, and the
full triage-aware agent — and reports keyword accuracy, intent accuracy, refusal correctness,
source-grounding rate, and latency for each.

```
python phase_four/evaluation/run_eval.py --model /path/to/Phi-3-mini-4k-instruct-q4.gguf
```

Optional flags:
```
--limit N            # only run the first N test cases (fast smoke test)
--configs a,b,c       # only run specific configs, e.g. --configs baseline_simple_rag
--db <path>           # defaults to data/lancedb
```

Results are written to `phase_four/evaluation/results/raw_results.json` and `summary.csv`, and a
summary table is printed to the terminal.

---

## Quick reference — all commands in one place

| Task | Command |
|---|---|
| Create venv (Windows) | `python -m venv venv` |
| Create venv (macOS/Linux) | `python3 -m venv venv` |
| Activate venv (Windows cmd) | `venv\Scripts\activate.bat` |
| Activate venv (Windows PowerShell) | `venv\Scripts\Activate.ps1` |
| Activate venv (macOS/Linux) | `source venv/bin/activate` |
| Install dependencies | `pip install -r requirements.txt` |
| Build/update knowledge base | `python phase_one/ingest.py` |
| Test retrieval quality | `python phase_one/ingest.py --test` |
| Run full agent (main entry point) | `python phase_three/chat.py --model <path>` |
| Run plain RAG (no agent) | `python phase_two/query.py --model <path>` |
| One-off sync check | `python sync/sync_manager.py --once` |
| Continuous background sync | `python sync/sync_manager.py --daemon` |
| Run the evaluation harness | `python phase_four/evaluation/run_eval.py --model <path>` |

---

## Known limitations

- **Some official sources block scripted downloads.** CDC and WHO's IRIS repository, among others,
  use bot-detection that rejects `sync_manager.py`'s requests even with a browser-like User-Agent —
  a real constraint on which sources can be auto-synced, not a bug in this project.
- **Query typo-correction is narrow by design** (`shared/query_normalizer.py`). It only guards
  against misspelled hazard-category terms (landslide, earthquake, bleeding, etc.), appends
  candidate corrections rather than replacing the original word (to avoid confidently "correcting"
  an ambiguous typo to the wrong term), and is not a general-purpose spell checker.
- **Evaluation metrics are lexical proxies, not ground truth.** `phase_four/evaluation/metrics.py`
  scores answers by keyword overlap and refusal-phrase detection, not human-verified correctness —
  useful for catching regressions and producing a first comparison table, but a sample of
  `raw_results.json` should be read manually before citing the numbers as final.
- This is a research prototype, not a certified emergency system. In any active, life-threatening
  emergency, contact local emergency services directly.
