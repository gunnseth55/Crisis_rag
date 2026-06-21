"""
phase2/rag_pipeline.py — RAG pipeline, cleaned up before Phase 3

Changes from previous version:
- MIN_SCORE raised from 0.35 to 0.45
  Scores below 0.45 mean the retrieved content is too loosely related.
  Better to say "I don't have that" than to answer from wrong context.

- query_with_history disabled — the previous implementation was prepending
  the last answer as text, which confused retrieval. The agent in Phase 3
  will handle conversation context properly through the ReAct loop.
  For now, every question is treated independently.

- Added source score logging so you can see retrieval quality at a glance.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from phase_one.embedder import Embedder
from phase_one.vector_store import VectorStore
from phase_two.llm import LLM


MIN_SCORE       = 0.45   # raised — below this, content is too loosely related
TOP_K           = 3
MAX_CHUNK_CHARS = 400


PROMPT_TEMPLATE = """Answer using ONLY the CONTEXT below.
If the answer is not clearly present, say: "I don't have specific guidance on that. Please call 112 for emergency help."

CONTEXT:
{context}

QUESTION: {question}

ANSWER (give numbered steps where possible):"""


class RAGPipeline:

    def __init__(self, db_path: str, model_path: str):
        print("[RAGPipeline] Initialising...")
        self.embedder     = Embedder()
        self.vector_store = VectorStore(db_path)
        self.vector_store.init()
        self.llm          = LLM(model_path)
        print(f"[RAGPipeline] Ready. {self.vector_store.count()} chunks in knowledge base.")

    def query(self, question: str) -> dict:
        print(f"\n[RAG] Query: {question}")

        # 1. Embed the question
        query_vector = self.embedder.embed(question)

        # 2. Retrieve top-K chunks
        results = self.vector_store.search(query_vector, top_k=TOP_K)

        # 3. Log scores so you can monitor retrieval quality
        print(f"[RAG] Retrieval scores:")
        for r in results:
            quality = "GOOD" if r["score"] >= 0.55 else "OK" if r["score"] >= 0.45 else "WEAK"
            print(f"      [{quality}] {r['source']}  score={r['score']:.3f}")

        # 4. Filter — only use chunks above MIN_SCORE
        relevant = [r for r in results if r["score"] >= MIN_SCORE]

        if not relevant:
            print("[RAG] No relevant chunks above threshold — using fallback response")
            return {
                "answer": (
                    "I don't have specific information about that situation "
                    "in my knowledge base.\n\n"
                    "Please call 112 immediately for emergency assistance, "
                    "or contact your local disaster management authority."
                ),
                "sources":  [],
                "question": question,
            }

        # 5. Build augmented prompt
        blocks = []
        for i, chunk in enumerate(relevant, 1):
            text = chunk["text"][:MAX_CHUNK_CHARS]
            blocks.append(f'[{i}] {chunk["source"]}:\n{text}')

        prompt = PROMPT_TEMPLATE.format(
            context="\n\n".join(blocks),
            question=question,
        )

        # 6. Generate answer
        print("[RAG] Generating answer...")
        answer = self.llm.generate(prompt)

        return {
            "answer":   answer,
            "sources":  [
                {"source": r["source"], "score": r["score"]}
                for r in relevant
            ],
            "question": question,
        }

    def query_with_history(self, question: str, history: list) -> dict:
        """
        NOTE: History-aware querying is intentionally simplified here.
        The ReAct agent in Phase 3 will handle multi-turn conversation
        properly. For Phase 2, we treat each question independently
        to avoid retrieval contamination from previous answers.
        """
        return self.query(question)