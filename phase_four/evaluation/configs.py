"""
evaluation/configs.py — the three system configurations compared in Phase 4
 
    1. baseline  — Simple RAG, no agent, no triage classifier.
                   phase_two.rag_pipeline.RAGPipeline, single retrieval pass.
 
    2. ablation  — RAG + agent (ReAct loop, reformulation-on-miss), but no
                   triage classifier. Every query uses the single generic
                   GENERAL config: no intent-specific thresholds, no
                   intent-specific system prompt, no EMOTIONAL short-circuit.
                   This isolates what the triage classifier itself buys you.
 
    3. full      — RAG + agent + triage classifier (the proposed system).
                   phase_three.agent.CrisisAgent(use_triage=True).
 
Each wrapper exposes .run(query: str) -> dict with a common shape:
    {
        "answer":     str,
        "sources":    list[{"source": str, "score": float}],
        "intent":     str | None,   # None for baseline
        "confidence": str | None,
        "iterations": int | None,   # None for baseline (single-pass)
    }
"""
 
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from phase_two.rag_pipeline import RAGPipeline
from phase_three.agent import CrisisAgent

class BaselineConfig:
    """Config 1 — simple RAG, no agent, no triage."""
    name="baseline_simple_rag"
    def __init__(self, db_path:str, model_path:str):
        self.pipeline=RAGPipeline(db_path=db_path, model_path=model_path)

    def run(self, query:str)->dict:
        result=self.pipeline.query(query)
        return{
            "answer": result["answer"],
            "sources": result["sources"],
            "intent": None,
            "confidence": None,
            "iterations": None,
        }

class AblationConfig:
    """ config 2 - agent+ReAct loop, triage classifier disabled"""
    name="ablation_agent_no_triage"

    def __init__(self, db_path:str, model_path:str):
        self.agent=CrisisAgent(db_path=db_path, model_path=model_path, use_triage=False)

    def run(self, query:str)->dict:
        response=self.agent.run(query)
        return {
            "answer": response.answer,
            "sources": response.sources,
            "intent": response.intent,
            "confidence": response.confidence,
            "iterations": response.iterations,
        }


class FullConfig:
    """Config 3 — agent + ReAct loop + triage classifier (proposed system)."""
    name="full_agent_with_triage"
    def __init__(self, db_path: str, model_path: str):
        self.agent = CrisisAgent(db_path=db_path, model_path=model_path, use_triage=True)
 
    def run(self, query: str) -> dict:
        response = self.agent.run(query)
        return {
            "answer": response.answer,
            "sources": response.sources,
            "intent": response.intent,
            "confidence": response.confidence,
            "iterations": response.iterations,
        }
    
def build_configs(db_path:str, model_path:str)->dict:
    """
    Builds all three configs. Each one loads its own LLM instance —
    this is intentionally simple/slow rather than sharing state, since
    for the harness correctness matters more than eval runtime.
    """
    configs={}
    
    for cls in (BaselineConfig, AblationConfig,FullConfig):
        print(f"\n{'='*60}\nBuilding config: {cls.name}\n{'='*60}")
        t0=time.time()
        configs[cls.name] = cls(db_path=db_path, model_path=model_path)
        print(f"[{cls.name}] built in {time.time() - t0:.1f}s")
    return configs