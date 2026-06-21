import sys
sys.path.insert(0, '.')

# Try to execute the llm.py file
with open('phase_two/llm.py') as f:
    code = f.read()

namespace = {}
try:
    exec(code, namespace)
    print("Execution successful")
    print("Namespace keys:", list(namespace.keys()))
    print("Has LLM class?", 'LLM' in namespace)
except Exception as e:
    print(f"Execution error: {e}")
    import traceback
    traceback.print_exc()
