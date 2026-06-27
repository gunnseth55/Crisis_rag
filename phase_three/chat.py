r"""
phase_three/chat.py — Crisis Assistant CLI (Phase 3)
 
HOW TO RUN:
    python phase_three/chat.py --model C:\Users\gunn\models\Phi-3-mini-4k-instruct-q4.gguf
 
WHAT'S NEW VS PHASE 2:
    - Every response is labelled with its intent: [MEDICAL], [EVACUATION] etc.
    - You can see the triage confidence: HIGH / MEDIUM / LOW
    - EMOTIONAL queries get a special compassionate response, no RAG
    - ReAct iterations are shown so you can see if reformulation was needed
    - Conversation history is stored cleanly without contaminating retrieval
"""
import sys
import argparse
import time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent.parent))
from phase_three.agent import CrisisAgent

DEFAULT_DB_PATH="data/lancedb"
INTENT_COLOURS = {
    "MEDICAL":    "\033[91m",   # red    — urgent
    "EVACUATION": "\033[93m",   # yellow — urgent
    "SURVIVAL":   "\033[96m",   # cyan   — important
    "EMOTIONAL":  "\033[95m",   # magenta — sensitive
    "GENERAL":    "\033[92m",   # green  — informational
}
RESET = "\033[0m"
BOLD  = "\033[1m"
def intent_badge(intent:str , confidence:str)->str:
    colour=INTENT_COLOURS.get(intent,"")
    return f"{colour}{BOLD}[{intent}]{RESET} (confidence:{confidence})"

def print_separator(char="-",width=60):
    print(char*width)

def run_chat(model_path:str, db_path:str):
    print("\n" + "="*60)
    print("  CRISIS MANAGEMENT AI ASSISTANT  —  Phase 3")
    print("  Intent-Aware Agentic RAG System")
    print("="*60)
    print(f"  Model: {Path(model_path).name}")
    print(f"  DB:    {db_path}")
    print("="*60)
    print("\nInitialising agent...")

    try:
        agent=CrisisAgent(db_path=db_path, model_path=model_path)
    except Exception as e:
        print(f"\n[ERROR] {e}")
        sys.exit(1)
    print("\nReady .Typr your question .Commands:'history','clear','quit' ")
    print_separator()

    conversation=[]
    while True:
        try:
            user_input=input("\nYou:").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\nSession ended. Stay safe.")
            break
        if not user_input:
            continue 

        if user_input.lower()=="quit":
            print("Session ended. Stay Safe!")
            break
        elif user_input.lower()=="clear":
            conversation=[]
            print("COnversation cleared")
            continue 
        elif user_input.lower()=="history":
            if not conversation:
                print("No conversation history yet")
            else:
                print_separator()
                for turn in conversation:
                    role="You" if turn["role"]=="user" else "Assistant"
                    intent=f"[{turn.get('intent','')}]" if turn["role"]=="assistant" else ""
                    print(f"{role}{intent}: {turn['content'][:200]}...")
                print_separator()
            continue 
        print("Classifying and searching")
        start=time.time()
        try:
            response = agent.run(user_input)
        except Exception as e:
            print(f"\n[ERROR] {e}")
            continue 
        elapsed=time.time()-start

        #display the response
        print_separator()
        print(intent_badge(response.intent, response.confidence))
        if response.iterations > 0:
            print(f"ReAct iterations: {response.iterations}")
        print()
        print(response.answer)

        if response.sources:
            print()
            print_separator("·")
            print("Sources:")
            for src in response.sources:
                score_pct = int(src["score"] * 100)
                print(f"  • {src['source']}  ({score_pct}% match)")
        print(f"\n({elapsed:.1f}s)")
        print_separator()
        conversation.append({"role": "user",      "content": user_input})
        conversation.append({"role": "assistant",  "content": response.answer,
                              "intent": response.intent})
        if len(conversation) > 20:
            conversation = conversation[-20:]

def main():
    parser = argparse.ArgumentParser(description="Crisis RAG Agent")
    parser.add_argument("--model", required=True, help="Path to .gguf model file")
    parser.add_argument("--db",    default=DEFAULT_DB_PATH, help="Path to LanceDB folder")
    args = parser.parse_args()
    if not Path(args.model).exists():
        print(f"[ERROR] Model not found: {args.model}")
        sys.exit(1)
 
    if not Path(args.db).exists():
        print(f"[ERROR] Database not found: {args.db}")
        print("Run: python phase1/ingest.py")
        sys.exit(1)
 
    run_chat(model_path=args.model, db_path=args.db)
 
 
if __name__ == "__main__":
    main()
 

