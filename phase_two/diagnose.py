"""
phase2/diagnose.py — Run this BEFORE query.py to check your model

Usage:
    python phase2/diagnose.py --model path/to/your-model.gguf

This will:
1. Load the model
2. Run a simple test prompt with NO crisis context
3. Print exactly what the model outputs
4. Tell you if the model and prompt format are working correctly
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def diagnose(model_path: str):
    print(f"\nDiagnosing model: {Path(model_path).name}")
    print(f"File size: {Path(model_path).stat().st_size / 1e9:.2f} GB")

    try:
        from llama_cpp import Llama
    except ImportError:
        print("\n[ERROR] llama-cpp-python not installed.")
        print("Run: pip install llama-cpp-python")
        return

    print("\nLoading model (15-30 seconds)...")
    try:
        model = Llama(
            model_path=model_path,
            n_ctx=512,
            n_gpu_layers=0,
            verbose=False,
        )
    except Exception as e:
        print(f"\n[ERROR] Could not load model: {e}")
        return

    print("Model loaded. Running diagnostic tests...\n")

    # --- Test 1: Raw completion (no chat format) ---
    print("=" * 50)
    print("TEST 1: Raw completion")
    print("Prompt: 'The capital of India is'")
    out = model("The capital of India is", max_tokens=10, echo=False)
    print(f"Output: '{out['choices'][0]['text'].strip()}'")
    print("Expected: something like 'New Delhi'")
    print()

    # --- Test 2: ChatML format (what Phi-3 instruct uses) ---
    print("=" * 50)
    print("TEST 2: ChatML format (Phi-3 Instruct)")
    chatml = "<|system|>\nYou are a helpful assistant.<|end|>\n<|user|>\nSay only the word: HELLO<|end|>\n<|assistant|>\n"
    out = model(chatml, max_tokens=20, echo=False, stop=["<|end|>", "<|user|>"])
    print(f"Output: '{out['choices'][0]['text'].strip()}'")
    print("Expected: 'HELLO' or 'Hello'")
    print()

    # --- Test 3: Llama 2 / Mistral format ---
    print("=" * 50)
    print("TEST 3: Llama2/Mistral format")
    llama2 = "[INST] Say only the word: HELLO [/INST]"
    out = model(llama2, max_tokens=20, echo=False, stop=["[INST]", "</s>"])
    print(f"Output: '{out['choices'][0]['text'].strip()}'")
    print("Expected: 'HELLO' or 'Hello'")
    print()

    # --- Test 4: Simple instruct format ---
    print("=" * 50)
    print("TEST 4: Simple instruct format")
    simple = "### Instruction:\nSay only the word HELLO.\n### Response:\n"
    out = model(simple, max_tokens=20, echo=False, stop=["###"])
    print(f"Output: '{out['choices'][0]['text'].strip()}'")
    print("Expected: 'HELLO'")
    print()

    print("=" * 50)
    print("DIAGNOSIS: Look at which test produced clean output.")
    print("  Test 1 clean  → model loads fine, it's a format issue")
    print("  Test 2 clean  → your model IS Phi-3 instruct, use ChatML")
    print("  Test 3 clean  → your model is Llama2/Mistral, use [INST] format")
    print("  None clean    → model file may be corrupted or incomplete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    args = parser.parse_args()

    if not Path(args.model).exists():
        print(f"[ERROR] File not found: {args.model}")
        sys.exit(1)

    diagnose(args.model)