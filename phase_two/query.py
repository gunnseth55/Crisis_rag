
"""
phase_two/query.py — Interactive crisis assistant CLI
 
HOW TO RUN:
    python phase_two/query.py --model path/to/phi3-mini.gguf
 
    Example (Windows):
        python phase_two/query.py --model C:/models/Phi-3-mini-4k-instruct-q4.gguf
 
    Example (Mac/Linux):
        python phase_two/query.py --model ~/models/Phi-3-mini-4k-instruct-q4.gguf
 
SPECIAL COMMANDS:
    sources     — show the full text of the last retrieved chunks
    history     — show your conversation so far
    clear       — start a new conversation (clears history)
    quit        — exit
 
DOWNLOAD THE MODEL:
    Phi-3 Mini Q4_K_M (2.2 GB):
    https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf/resolve/main/Phi-3-mini-4k-instruct-q4.gguf
 
    You can download with wget:
        wget https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf/resolve/main/Phi-3-mini-4k-instruct-q4.gguf
 
    Or just open the URL in your browser and it will download.
"""
 
import sys
import argparse
import time
from pathlib import Path
 
sys.path.insert(0, str(Path(__file__).parent.parent))
 
from phase_two.rag_pipeline import RAGPipeline
 
 
# Default path to the vector database created in Phase 1
DEFAULT_DB_PATH = "data/lancedb"
 
 
def print_separator(char="─", width=60):
    print(char * width)
 
 
def print_answer(result: dict, show_sources: bool = True):
    """Pretty-print a RAG result to the terminal."""
    print_separator()
    print("\nAssistant:")
    print(result["answer"])
 
    if show_sources and result["sources"]:
        print()
        print_separator("·")
        print(f"Sources used ({len(result['sources'])} chunks retrieved):")
        for i, src in enumerate(result["sources"], 1):
            score_pct = int(src["score"] * 100)
            print(f"  [{i}] {src['source']}  (relevance: {score_pct}%)")
    elif not result["sources"]:
        print()
        print("  [No relevant sources found in knowledge base]")
 
    print_separator()
 
 
def run_cli(model_path: str, db_path: str):
    """
    Main interactive loop.
    """
    print("\n" + "="*60)
    print("  CRISIS MANAGEMENT AI ASSISTANT")
    print("  Offline RAG System — Phase 2")
    print("="*60)
    print(f"  Model:    {Path(model_path).name}")
    print(f"  Database: {db_path}")
    print("="*60)
    print("\nInitialising system...")
 
    # Initialise the pipeline — loads embedder + LanceDB + LLM
    # This takes 10-20 seconds on first run due to model loading
    try:
        pipeline = RAGPipeline(db_path=db_path, model_path=model_path)
    except Exception as e:
        print(f"\n[ERROR] Failed to initialise: {e}")
        print("\nCommon fixes:")
        print("  - Check that the model path is correct")
        print("  - Make sure data/lancedb/ exists (run phase1/ingest.py first)")
        print("  - Ensure llama-cpp-python is installed")
        sys.exit(1)
 
    print("\nSystem ready. Type your question or 'quit' to exit.")
    print("Commands: 'sources' (show last chunks), 'clear' (new session), 'quit'")
    print_separator()
 
    conversation_history = []
    last_result = None
 
    while True:
        # Get user input
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\nSession ended.")
            break
 
        if not user_input:
            continue
 
        # ── Special commands ──────────────────────────────────────────────
 
        if user_input.lower() == "quit":
            print("Session ended. Stay safe.")
            break
 
        elif user_input.lower() == "clear":
            conversation_history = []
            last_result = None
            print("Conversation cleared. Starting fresh.")
            continue
 
        elif user_input.lower() == "history":
            if not conversation_history:
                print("No conversation history yet.")
            else:
                print_separator()
                for turn in conversation_history:
                    role = "You" if turn["role"] == "user" else "Assistant"
                    print(f"{role}: {turn['content'][:200]}...")
                print_separator()
            continue
 
        elif user_input.lower() == "sources":
            # Show the full text of the chunks from the last query
            if not last_result or not last_result["sources"]:
                print("No sources from last query. Ask a question first.")
            else:
                print_separator()
                print("Full source chunks from last query:")
                for i, src in enumerate(last_result["sources"], 1):
                    print(f"\n[{i}] {src['source']} (score: {src['score']:.3f})")
                    print_separator("·")
                    # Retrieve full text by re-querying
                    print(src["text"])
                print_separator()
            continue
 
        # ── Normal query ──────────────────────────────────────────────────
 
        start_time = time.time()
        print("Searching knowledge base and generating answer...")
 
        try:
            # Use conversation history for context on follow-up questions
            if conversation_history:
                result = pipeline.query_with_history(
                    user_input,
                    conversation_history
                )
            else:
                result = pipeline.query(user_input)
 
        except Exception as e:
            print(f"\n[ERROR] Query failed: {e}")
            print("Try asking a different question or type 'quit' to exit.")
            continue
 
        elapsed = time.time() - start_time
        print(f"(Generated in {elapsed:.1f}s)")
 
        # Display the answer
        print_answer(result)
 
        # Store in conversation history for follow-up context
        conversation_history.append({"role": "user", "content": user_input})
        conversation_history.append({"role": "assistant", "content": result["answer"]})
 
        # Keep only last 6 turns to avoid context overflow
        if len(conversation_history) > 12:
            conversation_history = conversation_history[-12:]
 
        last_result = result
 
 
def main():
    parser = argparse.ArgumentParser(
        description="Crisis Management RAG Assistant — Phase 2"
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Path to the Phi-3 Mini .gguf model file"
    )
    parser.add_argument(
        "--db",
        default=DEFAULT_DB_PATH,
        help=f"Path to LanceDB folder (default: {DEFAULT_DB_PATH})"
    )
 
    args = parser.parse_args()
 
    # Validate inputs before starting
    if not Path(args.model).exists():
        print(f"\n[ERROR] Model file not found: {args.model}")
        print("\nDownload Phi-3 Mini Q4_K_M (~2.2 GB) from:")
        print("  https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf")
        print("\nThen run:")
        print(f"  python phase_two/query.py --model /path/to/Phi-3-mini-4k-instruct-q4.gguf")
        sys.exit(1)
 
    if not Path(args.db).exists():
        print(f"\n[ERROR] Database not found: {args.db}")
        print("Run Phase 1 first:")
        print("  python phase1/ingest.py")
        sys.exit(1)
 
    run_cli(model_path=args.model, db_path=args.db)
 
 
if __name__ == "__main__":
    main()
 