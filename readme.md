# Project ResQ: Offline-First Agentic RAG for Crisis Response

An offline-first, resource-efficient Retrieval-Augmented Generation (RAG) agent designed for deployment in disaster zones and communication-degraded environments. The system operates entirely on-device (zero-server, zero-internet dependency) utilizing a Small Language Model (SLM) and a local vector database, with opportunistic synchronization when a network connection is available.

### 🔬 Core Research Contribution
Unlike standard RAG architectures or static agentic loops, this project introduces a **Pre-Retrieval Triage Intent Classifier** integrated into a ReAct (Reason+Act) loop. Before the retrieval mechanism triggers, the user's input is classified into domain-specific crisis categories (e.g., *Medical Emergency, Evacuation/Navigation, Survival/Resources*). This dynamically filters the vector search space and shifts the system's behavioral persona, significantly reducing chunk dilution, lowering execution latency, and eliminating hallucinatory bleed across unrelated crisis protocols.

---

## 🏗️ Project Architecture & Phase Blueprint

```text
                     [ User Query ]
                           │
                           ▼
             ┌───────────────────────────┐
             │  Triage Intent Classifier │ (Research Novelty)
             └─────────────┬─────────────┘
                           │
            [Intent Category & Dynamic Prompt]
                           │
                           ▼
             ┌───────────────────────────┐
  ┌─────────►│     Agent ReAct Loop      │◄────────┐
  │          │   (Reason -> Act -> Obs)  │         │
  │          └─────────────┬─────────────┘         │
  │                        │                       │
  │                  [Selects Tool]                │
  │                        │                       │
  │         ┌──────────────┴──────────────┐        │
  │         ▼                             ▼        │
  │   (Offline Mode)                (Online Mode)  │
  │  ┌──────────────┐              ┌────────────┐  │
  │  │ Local RAG    │              │ Web Search │  │
  │  │ (LanceDB)    │              │ API        │  │
  │  └──────┬───────┘              └─────┬──────┘  │
  │         │ [Filtered Chunks]          │         │
  └─────────┴────────────────────────────┴─────────┘
                           │
                 [Max Iterations / Stop]
                           │
                           ▼
                   [Final Answer]


