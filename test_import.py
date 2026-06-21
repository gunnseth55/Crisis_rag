import sys
sys.path.insert(0, '.')
try:
    from phase_two import llm
    print("Module imported successfully")
    print("Module contents:", dir(llm))
    print("Has LLM?", hasattr(llm, 'LLM'))
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
