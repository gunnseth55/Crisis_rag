import sys
sys.path.insert(0, '.')

code_to_test = """from llama_cpp import Llama

SYSTEM_PROMPT = "test"

class LLM:
    pass
"""

namespace = {}
try:
    exec(code_to_test, namespace)
    print("Execution successful")
    print("Has LLM?", 'LLM' in namespace)
    print("Keys:", [k for k in namespace.keys() if not k.startswith('__')])
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
