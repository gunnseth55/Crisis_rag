# Crisis RAG: presentation and demo notes

Reviewed against the code and local files on 14 October 2026.

## A 60-second introduction

"Crisis RAG is an offline-first assistant for disaster and first-aid questions.
We store emergency documents on the device and use retrieval-augmented
generation, or RAG, to find relevant passages before asking a local language
model to answer. This means the model receives evidence from our documents
instead of relying only on what it learned during training.

Our agent layer first identifies the kind of question: medical, evacuation,
survival, emotional, or general. It uses that label to choose a retrieval
threshold and a response prompt. If retrieval fails, it changes the search
query and tries once more. If both attempts fail, it returns a fallback.

The project uses MiniLM embeddings, LanceDB storage, and a quantized Phi-3
Mini model running through llama.cpp. It also includes optional background
document sync and an evaluation harness. It is a research prototype, and
the current agent is a bounded workflow with rule-based decisions."

## The two pipelines

### Preparing the knowledge base (Phase 1)

```text
PDF / TXT / Markdown documents
  -> extract text
  -> clean and split into overlapping chunks
  -> embed each chunk into a 384-number vector
  -> save vectors, text, and source names in local LanceDB
```

- `phase_one/ingest.py`: reads files in `knowledge_base/`; uses PyMuPDF for PDFs.
- `phase_one/chunker.py`: targets about 500 estimated tokens per chunk,
  with 64 estimated tokens of overlap. Token counts use characters / 4,
  rather than the model's tokenizer. Short chunks below 40 tokens are
  filtered on the long-document path.
- `phase_one/embedder.py`: uses `sentence-transformers/all-MiniLM-L6-v2`.
  Document and query vectors use the same model and are normalized.
- `phase_one/vector_store.py`: stores the vectors and original chunk text.
- Existing sources are skipped by filename stem. Editing an already indexed
  file alone does not update its stored chunks.

Local database observed during review: **2,740 chunks from 37 sources**.
That includes a GitHub README used for sync testing, which is unrelated to
emergency guidance.

### Answering a question (Phases 2 and 3)

```text
User question
  -> normalize possible hazard-word typos
  -> hybrid keyword / local-model intent classification
  -> choose intent-specific settings
  -> embed query and retrieve top 3 chunks
  -> filter chunks by score
       enough matches: build context prompt -> local Phi-3 -> answer
       no matches: add intent-specific search terms -> retry once
       still no matches: return fallback without generating an answer
```

Emotional intent takes a separate path: a prewritten supportive response,
with no retrieval or answer generation.

The current prompt includes only the **first 400 characters of each selected
chunk**. The model sees an excerpt, not the whole document or even the whole
chunk. This can omit useful instructions and is an area for improvement.

## Exactly where the agentic AI was added

Phase 2, `phase_two/rag_pipeline.py`, provides the baseline:

```text
One question -> one retrieval -> generation or fallback
```

Phase 3 adds two components:

1. `phase_three/triage_classifier.py`: `TriageClassifier.classify()` decides
   the intent. Strong medical keywords take priority. Strong keyword
   matches bypass model inference; weaker or absent matches use the local
   LLM. The high/medium/low confidence labels are rule labels, not calibrated
   probabilities.
2. `phase_three/agent.py`: `CrisisAgent.run()` selects the policy;
   `_react_loop()` retrieves, checks the threshold, and controls the retry;
   `_reformulate()` adds fixed terms such as "first aid treatment";
   `_generate()` supplies the selected excerpts and intent-specific prompt.

The agent returns an `AgentResponse` containing the intent, confidence,
answer, sources and scores, and number of retrieval iterations.

| Intent | Minimum retrieval score | Answer token budget | Behavior |
|---|---:|---:|---|
| MEDICAL | 0.50 | 350 | Stricter retrieval and first-aid prompt |
| EVACUATION | 0.45 | 300 | Ordered evacuation guidance prompt |
| SURVIVAL | 0.45 | 350 | Practical survival guidance prompt |
| GENERAL | 0.40 | 300 | General crisis-information prompt |
| EMOTIONAL | Not used | Not used | Fixed supportive response, zero searches |

The plain Phase 2 baseline has a threshold of 0.45.

## How to explain "agentic" accurately

Say: **"We implemented a lightweight, ReAct-inspired agent workflow with
intent routing, retrieval feedback, and a bounded retry."**

Its actions are document searches. Its observation is the returned retrieval
scores. Its next step is decided by code: generate, reformulate, or stop.
Query reformulation uses templates, not LLM-generated reasoning.

It has no open-ended planning, external rescue-service tool, autonomous
dispatch, or multiple cooperating agents. It selects one intent for a query;
mixed medical/evacuation questions are not decomposed into separate tasks.
No LangChain or LangGraph framework is used: the workflow is ordinary Python.

Chat history is saved for display in the CLI and Streamlit UI, but previous
turns are not passed into retrieval or generation. Ask complete questions
during the demo rather than relying on follow-ups like "what about that?".

## A concrete example to narrate

For "My leg is bleeding and it won't stop":

1. The bleeding keyword selects MEDICAL before model classification is needed.
2. The agent selects the 0.50 threshold and first-aid response prompt.
3. MiniLM embeds the query; LanceDB returns the top three matching chunks.
4. Chunks meeting the threshold become context for local Phi-3.
5. The UI shows the answer, MEDICAL label, confidence, sources, retrieval
   scores, latency, and number of searches.
6. If no chunk meets the threshold, the second search adds "first aid
   treatment" to the query. If that also fails, the agent uses its fallback.

This describes the code path; it is not a guarantee of the exact answer or
that any particular question will produce a second search.

## Models and retrieval scores

- MiniLM is the encoder: it converts text into vectors for semantic search.
- Phi-3 Mini is the generator: it converts the question and context into
  a natural-language answer. The local GGUF file is quantized to reduce
  memory and storage needs. The wrapper uses a 4096-token context and CPU
  execution (`n_gpu_layers=0`, four threads).
- No training or fine-tuning is implemented here. Ingestion builds the index;
  it does not train Phi-3.
- The score is `1 / (1 + returned L2 distance)`. It is a transformed distance,
  not cosine similarity, a percentage of correctness, or an emergency risk
  score. The thresholds are heuristic.
- Sources identify retrieved evidence. Source presence alone does not prove
  that every generated claim follows from that evidence.

## Optional sync

`sync/sync_manager.py` starts a background thread in the CLI and UI. Every
30 minutes it checks connectivity, downloads configured documents, compares
SHA-256 checksums, and re-embeds changed documents.

**Current configuration:** `sync/sources.json` points to the GitHub gitignore
README as a test source. Do not say that WHO/CDC/NDMA guidance is being
automatically updated in the current demo. Verified document download URLs
must be configured for that use case.

Remaining sync issues: an unextractable document can be reported as updated
without new chunks; files are overwritten before validation; deleting and
then adding chunks is not an atomic replacement. Sync reliability needs
more work before production use.

## Saved evaluation results

The repository contains results from **18 July 2026**, on 24 cases:

| Configuration | Keyword score | Intent accuracy | Refusal correctness | Source presence | Mean latency |
|---|---:|---:|---:|---:|---:|
| Plain RAG | 0.566 | N/A | 1.000 | 0.875 | 14.556 s |
| Agent without triage | 0.531 | N/A | 0.917 | 1.000 | 12.394 s |
| Full agent with triage | 0.778 | 0.905 | 0.958 | 0.875 | 20.104 s |

Use these as **historical prototype results**. They were not regenerated
after this review's code changes. Keyword overlap and source presence are
proxy metrics, not clinical correctness or verified factual faithfulness.
Intent accuracy excludes cases marked not applicable. The full agent's saved
results misclassify two general information/preparation questions as
EVACUATION.

The full agent has a higher saved keyword score and higher latency. It does
not win every metric. Also, the agent-without-triage configuration uses a
0.40 threshold and a different prompt from the baseline's 0.45 threshold,
so that comparison does not isolate the retry loop alone.

## Demo commands

Run from the project root in PowerShell:

```powershell
python -m pytest tests -q
python -m streamlit run ui/app.py
```

Enter this existing model path in the UI:

```text
C:\Users\gunn\models\Phi-3-mini-4k-instruct-q4.gguf
```

Keep the database path as `data/lancedb`, then load the assistant.

CLI alternative:

```powershell
python phase_three/chat.py --model C:/Users/gunn/models/Phi-3-mini-4k-instruct-q4.gguf
```

Suggested complete questions, tried beforehand on your machine:

1. "How can I purify drinking water during a disaster?"
2. "My leg is bleeding a lot and it won't stop."
3. "Water is rising and I need to evacuate and escape now."
4. "I am scared and alone right now."

Show the intent badge, source scores, and iteration count. Emotional support
has zero iterations because it uses a fixed response. Do not promise that
an out-of-scope question will always be refused: similarity thresholds and
the model's prompt compliance are imperfect.

Initial model setup needs downloads. Cached embeddings, the local GGUF file,
and the vector database are needed for offline answering. The embedding
loader now attempts local-only loading first to avoid network metadata
requests when weights are cached.

## Fixes made during this review

- Keyword matching now starts at word boundaries, preventing "weather"
  from matching "eat"/"heat" inside the word.
- Fuzzy correction is more conservative, preventing "France" from adding
  "fracture", while retaining explicit observed hazard typos.
- Strong keyword decisions skip unnecessary classifier-model inference.
- Classification explanation strings use console-safe ASCII punctuation,
  fixing an observed Windows `UnicodeEncodeError` on an arrow character.
- Embedding loading tries the local cache without network requests first.
- CLI/UI source scores are labelled as retrieval scores rather than
  misleading match percentages.
- The agent's source-list type and CLI ingestion command were corrected.
- Regression coverage was added; 39 tests passed after the classifier fixes.

Verification completed: all 39 tests pass, Python source syntax parses, and
`git diff --check` is clean. A real SURVIVAL query completed retrieval and
generation. With `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1`, a MEDICAL
query completed with two sources and one search; an EMOTIONAL query returned
the fixed response with zero sources and zero searches. This checks the
execution paths, not the factual correctness of generated advice. The full
24-case evaluation and visual Streamlit UI inspection were not rerun.

## Questions you may get

**Why RAG?** It supplies domain documents at inference time and makes retrieved
evidence inspectable without retraining the model.

**Why offline?** Disaster connectivity can be unreliable. Answering uses local
models and a local index after setup; sync is optional.

**What is novel here?** The combination of offline operation, intent-specific
retrieval/prompting, medical keyword priority, and a bounded retrieval retry.
Avoid claiming that these ingredients individually are new research.

**Does the agent know the safest route?** It can retrieve general evacuation
guidance. There is no live map, location, road-closure feed, or route planner.

**Does it call emergency services?** No. It can include a contact instruction
in its text. There is no calling or dispatch integration.

**Does it understand all typos or languages?** No. Typo handling is narrow and
heuristic, and the prompts/keyword rules are English-oriented.

**What should improve next?** Better intent handling for informational,
negated, and mixed emergencies; token-budgeted context instead of 400-character
excerpts; human-reviewed answer evaluation; transactional sync; and proper
conversation context.
