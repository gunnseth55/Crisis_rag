"""
intent aware ReAct agent
PURPOSE:
    classify intent
    configures retrieval and prompting based on intent
    runs a lightweight react loop (reason->act->observe)
    return a structured response with the intent label and sources

    ReAct: query->think->retrieve->observe->think->act
     The "think" step allows the agent to:
    - Decide if retrieved context is sufficient or needs a different query
    - Handle multi-part questions ("I have a wound AND need to evacuate")
    - Know when to stop and say "I don't have that" vs guess
    INTENT-SPECIFIC CONFIGURATIONS:
    Each intent gets its own:
    - min_score:      minimum retrieval score to use a chunk
    - system_prompt:  tailored instruction to the model
    - max_tokens:     response length budget

    (MEDICAL gets the strictest min_score (0.50) because incorrect medical
    advice can cause physical harm. Better to say "call 112" than to give
    wrong first aid steps from loosely related context.
 
    EMOTIONAL skips retrieval entirely. No document in your knowledge base
    contains the right content for "I'm terrified and alone." Sending it
    through RAG would return first aid steps — completely wrong. The agent
    handles this with a pre-written calm response + 112 reference.)

    LEARNING RESOURCES:
    ReAct paper: https://arxiv.org/abs/2210.03629
    (Read sections 2 and 3 — the algorithm is simple, 2 pages)
 
    Why prompt engineering matters for crisis contexts:
    "Prompt Engineering Guide" — https://www.promptingguide.ai/
    (Free, comprehensive, read the "Zero-Shot" and "Chain-of-Thought" sections)
"""

import sys
from pathlib import Path
from dataclasses import dataclass 
sys.path.insert(0,str (Path(__file__).parent.parent))

from phase_one.embedder import Embedder
from phase_one.vector_store import VectorStore
from phase_two.llm import LLM
from phase_three.triage_classifier import TriageClassifier, TriageResults

INTENT_CONFIGS={
     "MEDICAL": {
        "min_score":   0.50,
        "max_tokens":  350,
        "system_prompt": (
            "You are a crisis first aid assistant. A person needs urgent medical help.\n"
            "Answer ONLY from the CONTEXT. Give clear numbered steps.\n"
            "If information is not in the context, say: 'Call 112 immediately — do not wait.'\n"
            "Do NOT speculate. Incorrect medical advice is dangerous.\n"
            "Start your response with the most critical action first."
        ),
    },
     "EVACUATION": {
        "min_score":   0.45,
        "max_tokens":  300,
        "system_prompt": (
            "You are a crisis evacuation assistant. A person needs to escape danger NOW.\n"
            "Answer ONLY from the CONTEXT. Give numbered steps in order of urgency.\n"
            "Be direct and fast — every second counts.\n"
            "End with: 'Call 112 and tell them your location.'"
        ),
    },"SURVIVAL": {
        "min_score":   0.45,
        "max_tokens":  350,
        "system_prompt": (
            "You are a disaster survival assistant.\n"
            "Answer ONLY from the CONTEXT. Give practical, step-by-step instructions.\n"
            "Focus on what the person can do RIGHT NOW with limited resources.\n"
            "If not in context, say: 'I don't have that information. Call 112.'"
        ),
    },
     "EMOTIONAL": {
        "min_score":   0.0,   # not used — EMOTIONAL skips RAG
        "max_tokens":  200,
        "system_prompt": None,  # not used — EMOTIONAL has a fixed response
    },
    "GENERAL": {
        "min_score":   0.40,
        "max_tokens":  300,
        "system_prompt": (
            "You are a crisis information assistant.\n"
            "Answer ONLY from the CONTEXT provided.\n"
            "If not in context, say: 'I don't have that information. Call 112 for help.'\n"
            "Be clear and concise."
        ),
    },
}
EMOTIONAL_RESPONSE="""
I can hear that you're in a very frightening situation. That fear is completely understandable.
 
Right now, focus on one thing at a time:
1. Take a slow breath in for 4 counts, hold for 4, breathe out for 4.
2. Look around you — identify one safe thing you can see or touch.
3. You are not alone. Help is available.
 
Please call 112 now — tell them your location and that you need help.
Emergency services are trained to help you through exactly this situation.
 
If you have a specific physical danger (injury, fire, flood), tell me what it is and I will give you steps."""
 
# The RAG prompt template — kept short to control token count
RAG_PROMPT = """CONTEXT:
{context}
 
QUESTION: {question}
 
ANSWER:
"""

@dataclass
class AgentResponse:
    """
    Structured output from the agent.
    intent: classified intent label
    confidence: triage confidence ("high"/"medium"/"low")
    answer: the last generated response text
    sources:list of {"source": str, "score": float}
    iterations: how many ReAct iterations were used (1 or 2)
    """
    intent:str
    confidence:str
    answer:str
    sources:str
    iterations:int

class CrisisAgent:
   def __init__(self, db_path:str , model_path:str):
       print("[Agent] Initialising.........")
       self.embedder=Embedder()
       self.vector_store=VectorStore(db_path)
       self.vector_store.init()
       self.llm=LLM(model_path)
       self.classifier=TriageClassifier(self.llm)
       total=self.vector_store.count()
       print(f"[Agent] ready . {total} chunks in knowledge base.")

   def run(self, query:str)->AgentResponse:
       print(f"\n [Agent] query :{query}")
       triage: TriageResults=self.classifier.classify(query)
       print(f"[Agent] Intent: {triage.intent}  Confidence: {triage.confidence}")
       print(f"[Agent] Reasoning : {triage.reasoning}")
       if triage.intent=="EMOTIONAL":
           return AgentResponse(
               intent="EMOTIONAL",
               confidence=triage.confidence,
               answer= EMOTIONAL_RESPONSE,
               sources=[],
               iterations=0,
           )
       config =INTENT_CONFIGS[triage.intent]
       answer, sources, iterations = self._react_loop(
            query      = query,
            config     = config,
            intent     = triage.intent,
        )
       return AgentResponse(
           intent     = triage.intent,
            confidence = triage.confidence,
            answer     = answer,
            sources    = sources,
            iterations = iterations,
       )
   def _react_loop(
                 self,
        query:   str,
        config:  dict,
        intent:  str,
    )->tuple[str, list, int]:
      """
        Lightweight ReAct loop — max 2 iterations.
 
        Iteration 1:
            - Search with the original query
            - Check if top score meets the intent's min_score threshold
            - If yes: generate answer → done
            - If no: reformulate query → try again
 
        Iteration 2:
            - Search with the reformulated query
            - Generate answer with whatever was retrieved
            - If still nothing useful: return fallback
        Returns: (answer_text, sources_list, iteration_count)
      """
      search_query=query
      min_score=config["min_score"]
      system_prompt  = config["system_prompt"]
      max_tokens     = config["max_tokens"]

      for iteration in range(1,3):
          print(f"[Agent] Iteration {iteration}: searching for '{search_query[:60]}'")
          query_vector = self.embedder.embed(search_query)
          results= self.vector_store.search(query_vector, top_k=3)
          for r in results:
             quality="GOOD" if r["score"]>=0.55 else "OK" if r["score"] >=min_score else "WEAK"
             print(f"[{quality}]{r['source']} source={r['score']:.3f}")
          relevant=[r for r in results if r["score"]>= min_score]
          if relevant:
                answer=self._generate(
                   query=query,
                   relevant=relevant,
                   system_prompt=system_prompt,
                   max_tokens=max_tokens
                )
                sources=[
                   {"source":r["source"], "score":r["score"]}
                   for r in relevant
                ]
                return answer, sources, iteration
          elif iteration==1: #nothing useful in the first try so reformulate
                print(f"[Agent] No relevant chunks above {min_score} — reformulating query")
                search_query = self._reformulate(query, intent)
          else: # second iteration also failed
                 print(f"[Agent] Retrieval failed after 2 iterations — using fallback")
                 fallback = (
                    f"I don't have specific information about that in my knowledge base.\n\n"
                    f"Please call 112 immediately for emergency assistance.\n"
                    f"Tell them your location and the nature of the emergency."
                )
                 return fallback, [], iteration
      return "Please call 112 for emergency help.", [], 2
   def _reformulate(self, original_query:str, intent:str)->str:
       """
        Reformulate the search query using the intent label as a guide.
        This is the "Reason" step of ReAct — the agent thinks about
        what terminology the knowledge base might use for this topic.
 
        We use simple template-based reformulation rather than asking the
        model to reformulate — faster and more predictable.
       """
       reformulations={
            "MEDICAL":    f"first aid treatment {original_query}",
            "EVACUATION": f"evacuation steps {original_query} emergency escape",
            "SURVIVAL":   f"survival guide {original_query} emergency",
            "GENERAL":    f"disaster safety {original_query}",
       }
       return reformulations.get(intent, f"emergency guide {original_query}")
   def _generate(self, query:str, relevant:list, system_prompt:str, max_tokens:int)->str:
       #build an augemnted prompt and generate an answer
       blocks=[]
       for i,chunk in enumerate(relevant,1):
           text=chunk["text"][:400]
           blocks.append(f'[{i}] {chunk["source"]}:\n{text}')
       prompt=RAG_PROMPT.format(
           context="\n\n".join(blocks),
           question=query,
       )
       #temporarily swap the system prompt to the intent-specific one
       original_system=self.llm._model
       full_prompt= (
            f"<|system|>\n{system_prompt}<|end|>\n"
            f"<|user|>\n{prompt}<|end|>\n"
            f"<|assistant|>\n"
        )
       estimated=len(full_prompt)//4
       print(f"[Agent] Generating answer (~{estimated} tokens)...")
       response = self.llm._model(
            full_prompt,
            max_tokens    = max_tokens,
            temperature   = 0.1,
            top_p         = 0.9,
            repeat_penalty= 1.1,
            stop          = ["<|end|>", "<|user|>", "<|endoftext|>"],
            echo          = False,
        )
 
       return response["choices"][0]["text"].strip()
       


 