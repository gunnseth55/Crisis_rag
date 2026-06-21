"""
phase2/llm.py — Phi-3 Mini Instruct, n_ctx=4096
"""

from llama_cpp import Llama
from pathlib import Path


SYSTEM_PROMPT = """You are a crisis assistance AI for natural disaster emergencies.
Answer ONLY using the CONTEXT provided. If not found, say "Call 112 immediately."
Be calm, direct, and give numbered steps. Keep answer under 150 words."""


class LLM:
    def __init__(self, model_path: str):
        print(f"[LLM] Loading: {Path(model_path).name}")
        self._model = Llama(
            model_path=model_path,
            n_ctx=4096,        # full training context — the warning was just informational
            n_gpu_layers=0,
            n_threads=4,
            verbose=False,
        )
        print("[LLM] Ready.")

    def generate(self, user_prompt: str, max_tokens: int = 300) -> str:
        full_prompt = (
            f"<|system|>\n{SYSTEM_PROMPT}<|end|>\n"
            f"<|user|>\n{user_prompt}<|end|>\n"
            f"<|assistant|>\n"
        )
        estimated = len(full_prompt) // 4
        print(f"[LLM] Prompt: ~{estimated} tokens")

        response = self._model(
            full_prompt,
            max_tokens=max_tokens,
            temperature=0.1,
            top_p=0.9,
            repeat_penalty=1.1,
            stop=["<|end|>", "<|user|>", "<|endoftext|>"],
            echo=False,
        )
        return response["choices"][0]["text"].strip()