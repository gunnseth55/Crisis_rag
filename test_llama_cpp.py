import sys
sys.path.insert(0, '.')

try:
    from llama_cpp import Llama
    print("llama_cpp import successful")
except Exception as e:
    print(f"llama_cpp import failed: {e}")
    import traceback
    traceback.print_exc()
