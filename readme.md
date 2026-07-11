
## Phase 1 — Knowledge base construction

This is your foundation. Every answer the system gives comes from here, so getting this right matters most.

**What you're doing:** You need to collect disaster-relevant documents — first aid procedures, evacuation protocols, WHO emergency guidelines, Red Cross survival manuals, NDMA (India) guidelines — and convert them into a vector database that lives entirely on disk, no internet needed at query time.

**The core concept — why a vector database:** Normal search finds documents by keyword overlap. Vector search finds documents by *semantic meaning*. If someone types "I can't breathe, there's smoke everywhere", keyword search fails unless the document literally contains those words. Vector search maps the query and the documents into the same numerical space, so "can't breathe, smoke" lands near "smoke inhalation treatment" and "respiratory emergency protocol" — because they mean the same thing in context. This is critical for disaster scenarios where victims type in fragmented, panicked language.

**The process:** You take each document, split it into ~500-token chunks (paragraphs roughly), run each chunk through an embedding model (a small neural network that converts text into a list of ~384 numbers), and store those numbers in LanceDB — a file-based vector database that requires zero server, zero internet, and runs on a laptop.

**Endpoints / deliverables at end of Phase 1:**
- A `data_ingestion.py` script that reads PDFs and text files and populates the DB
- A `knowledge_base/` folder containing the crisis documents you curated
- A `lancedb/` folder containing the vector database on disk
- A simple test: given query "someone is unconscious", retrieve top 5 chunks and verify they're about first aid for unconsciousness

**Concepts to understand before building:**
- What is an embedding? (text → fixed-size vector via a neural network)
- What is cosine similarity? (how we measure "how close" two vectors are)
- What is chunking and why overlap matters (context at boundaries)
- The `sentence-transformers` library and `all-MiniLM-L6-v2` model specifically

---

## Phase 2 — Offline RAG pipeline

Now you connect query → retrieval → answer, entirely locally, no internet.

**What you're doing:** A user types a question. You embed that question using the same model you used to embed documents (this symmetry is critical — both must live in the same vector space). You search the vector database for the 5 most relevant chunks. You build a prompt that says "answer only from this context" and pass it to a local SLM (small language model) running on the device via llama.cpp.

**The core concept — why RAG instead of just fine-tuning the model:** You cannot fine-tune a model on disaster protocols every time new guidelines are released. RAG separates the knowledge (the database, easy to update) from the reasoning (the model, expensive to change). When NDMA releases new flood protocols, you just run the ingestion script — no retraining.

**The local SLM choice:** For desktop use, Phi-3 Mini (3.8B, Q4_K_M quantized, ~2.2GB) is ideal. It fits in RAM on any modern laptop, runs at ~5-10 tokens/second on CPU (fast enough for a query), and follows instructions well. The model file is a single `.gguf` file that you download once. `llama-cpp-python` runs it — one Python library, no GPU required.

**The prompt structure — this is where hallucination is controlled:**
```
You are a crisis assistance system. Answer ONLY from the context below.
If the answer is not in the context, say "I don't have that information — 
please contact emergency services."

CONTEXT:
[1] From "WHO First Aid Manual": ...chunk text...
[2] From "NDMA Flood Guidelines": ...chunk text...

QUESTION: How do I treat someone who has swallowed floodwater?

ANSWER:
```

The "ONLY from context" instruction is the key to faithfulness. The model is forced to cite from what was retrieved rather than hallucinate from its training data.

**Endpoints / deliverables at end of Phase 2:**
- A `rag_pipeline.py` with `ingest()` and `query()` functions
- A working CLI: type a question, get an answer with source citations
- A test suite: 10 crisis questions, manually verify answers are grounded in retrieved context
- Latency measurement: log time from query to first token, time to full answer

---

## Phase 3 — Agentic AI core

This is what differentiates your work from a simple RAG chatbot, and where the novelty really sits.

**What you're doing:** Instead of always retrieves → answers, you add a reasoning layer. The agent receives a user query, thinks about what it needs, selects appropriate tools, and may run multiple steps before answering. In a disaster context, a query like "I'm trapped, my leg hurts, I have no water and there's a fire nearby" requires different tools than "what is the evacuation route for Chennai floods".

**The ReAct loop — Reason, Act, Observe:**
The model outputs a thought ("I need to check first aid for leg injuries and also fire safety"), then an action (call `search_knowledge_base` with "leg fracture immobilization"), then reads the observation (what the search returned), then reasons again, maybe calls another tool, and finally produces an answer. This is a loop with a maximum iteration cap (e.g. 6 rounds) to prevent runaway computation.

**The novelty: a triage intent classifier before the loop.** This is something the existing literature (including MobileRAG and the papers you cited) does not do. Before the ReAct loop even starts, you classify the intent of the query into one of a few crisis categories: medical emergency, evacuation/navigation, survival/resources, emotional support, information lookup. This does two things: it lets you route to domain-specific retrieval filters (a "medical emergency" query should only search medical protocol chunks, not evacuation maps), and it lets you adjust the system prompt accordingly (a panicked person asking about injuries needs a calmer, more direct tone than someone asking about evacuation routes). This small addition — intent-aware routing before RAG — is the research contribution you can articulate in your paper.

**The tools your agent has:**
- `search_knowledge_base(query)` — always available, offline
- `triage_classifier(text)` — classifies intent, always offline
- `web_search(query)` — only available when online, falls back gracefully
- `calculate(expression)` — offline, for things like "how many hours until sunrise"

**The connectivity-aware behavior:** When online, the agent can call `web_search` for real-time updates (current shelter locations, live flood maps). When offline, it gracefully falls back to the pre-loaded knowledge base with a note to the user that information may not reflect the latest situation. This is the "offline-first with opportunistic sync" architecture from your abstract.

**Endpoints / deliverables at end of Phase 3:**
- `agent/orchestrator.py` with the ReAct loop
- `agent/triage_classifier.py` with the intent classifier (this can be a small fine-tuned model or even a simple zero-shot classifier using the same SLM)
- `agent/tools.py` with the tool registry
- The agent runs correctly in both online and offline modes
- Test: run 5 multi-turn conversations simulating a disaster scenario

---

## Phase 4 — Evaluation and sync

This is where you produce the numbers that go into your paper.

**What you're doing:** You build a test suite of crisis scenarios. You run your system under three conditions: fully offline (no internet), partially degraded (internet available but slow), and fully online. You measure accuracy (does the answer match the expected answer?), faithfulness (is the answer grounded in retrieved context, not hallucinated?), and latency (how many seconds to get an answer?).

**The sync mechanism:** When internet is available, the system checks for updated documents from authoritative sources (WHO, NDMA, Red Cross) and re-ingests anything that changed (comparing checksums). This is the "synchronizes with outside sources when network is available" from your methodology. On a desktop, this can be as simple as a background thread that polls every 30 minutes when online.

**How you demonstrate novelty in the paper:** You compare three configurations:
1. Simple RAG (no agent, no triage) — baseline
2. RAG + agent (no triage classifier) — ablation
3. RAG + agent + triage classifier (your full system) — proposed

Show that Configuration 3 answers multi-part crisis queries more accurately and retrieves more relevant chunks than the baselines. That comparison is your contribution.

**Endpoints / deliverables at end of Phase 4:**
- `evaluation/test_cases.json` — 20-30 crisis scenarios with expected answers
- `evaluation/run_eval.py` — runs all test cases and outputs a results table
- `sync/sync_manager.py` — background sync of the knowledge base when online
- The final paper section: results table comparing the three configurations

---

The thing that will make reviewers take this seriously: you are not building a general-purpose chatbot. You are building something evaluated specifically on crisis-scenario performance, with a domain-specific pre-loaded knowledge base, and with an architecture that degrades gracefully rather than failing when the network drops. That specificity — grounded in the problem Nunavath, Park, and your other cited authors identified as the gap — is what makes this original work and not a demo.

When you're ready, tell me which phase to start with and we'll go deep on exactly what to code.