"""
every prompt sent to the model:
classify the emergency queries into one of these categories:
Categories: Medical, evacuation, survival, emotional , general

THE FIVE CATEGORIES AND WHAT CHANGES FOR EACH:
 
    MEDICAL
        What: injuries, first aid, CPR, unconscious, bleeding, fractures
        Retrieval: min_score raised to 0.50 (must be confident)
        Prompt: emphasises step-by-step precision, do not move injured
        Urgency: HIGH — wrong advice here can kill
 
    EVACUATION
        What: escape routes, where to go, trapped, rising water
        Retrieval: standard min_score 0.45
        Prompt: emphasises speed, numbered steps, call 112
        Urgency: HIGH — time-critical
 
    SURVIVAL
        What: water purification, shelter, warmth, food safety
        Retrieval: standard min_score 0.45
        Prompt: practical, step-by-step, resource-conscious
        Urgency: MEDIUM — important but not immediate
 
    EMOTIONAL
        What: fear, panic, hopelessness, trauma
        Retrieval: SKIPPED — emotional support doesn't come from documents
        Response: calm acknowledgement, breathing exercise, call 112
        Urgency: HIGH — but a different kind of urgency
        GENERAL
        What: information queries, preparation, general advice
        Retrieval: standard min_score 0.40 (can be more lenient)
        Prompt: informative, balanced
        Urgency: LOW

        
LEARNING RESOURCES:
    Zero-shot classification with LLMs:
    "Is GPT-3 a Good Few-Shot Learner?" — Brown et al., 2020
    https://arxiv.org/abs/2005.14165
    (The paper that showed LLMs can classify without task-specific training)
 
    Why triage matters in crisis contexts:
    The START triage system (Simple Triage and Rapid Treatment) from
    emergency medicine — same principle: classify first, allocate
    resources second. https://chemm.hhs.gov/startadult.htm
"""

import re
from dataclasses import dataclass

INTENTS={"MEDICAL","EVACUATION","SURVIVAL","EMOTIONAL","GENERAL"}

@dataclass
class TriageResults:
    """
    the output of the triage classifier 
    intent: one of the five categories strings
    confidence: "high","medium", or "low"- based on how clear the query was
    reasoning: breif explanation
    """
    intent:str
    confidence:str
    reasoning:str

CLASSIIFICATION_PROMPT="""Classify this emergency query into exactly one category.
MEDICAL: injuries, bleeding, CPR, unconscious, not breathing, pain, wound, fracture, burn, heart attack, choking
EVACUATION: escape, leave, run, where to go, route, trapped, flood rising, get out, which way
SURVIVAL:  water purification, food safety, warmth, shelter, cold weather, fire, overnight, resources
EMOTIONAL:  scared, afraid, panic, hopeless, crying, alone, traumatized, cannot cope, worried, stressed
GENERAL: what is, how does, explain, information, prepare, news, advice, general question

Query:"{query}"

Answer with ONE word only(MEDICAL/EVACUATION/SURVIVAL/EMOTIONAL/GENERAL):
"""

#keyword based fallback when the model isn't loaded or unavailable
KEYWORD_RULES=[
     ("MEDICAL", ["bleeding", "wound", "hurt", "injury", "injured", "pain",
                    "unconscious", "not breathing", "cpr", "broken", "fracture",
                    "burn", "cut", "blood", "heart", "choking", "swallowed",
                    "overdose", "poisoned", "breathing"]),
    ("EVACUATION",["evacuate", "evacuation", "escape", "leave", "run", "flee",
                    "trapped", "stuck", "where to go", "route", "shelter",
                    "rising water", "get out", "exit", "way out"]),
    ("SURVIVAL", ["water", "purify", "drink", "food", "eat", "warm", "cold",
                    "shelter", "fire", "overnight", "survive", "heat", "freeze",
                    "hypothermia", "frostbite", "dehydration"]),
    ("EMOTIONAL" ,["scared", "afraid", "fear", "panic", "hopeless", "crying",
                    "alone", "traumatized", "cannot cope", "worried", "anxiety",
                    "stressed", "help me", "i dont know what to do",
                    "i don't know what to do"]),
]

def _keyword_classify(query:str) ->str:
    """
    Fallback keyword based classifier .
    Returns the intent  with the most keywords matches, or GENERAL.
    """
    query_lower=query.lower()
    scores={}
    for intent, keywords in KEYWORD_RULES:
        score=sum(1 for kw in keywords if kw  in query_lower)
        if score>0:
            scores[intent]=score
    if not scores:
        return "GENERAL"
    return max(scores, key=scores.get)

class TriageClassifier:
    """
     Usage:
        classifier = TriageClassifier(llm_model)
        result = classifier.classify("my leg is bleeding badly")
        # result.intent    → "MEDICAL"
        # result.confidence → "high"
        # result.reasoning → "query contains 'bleeding' — medical emergency"

    """
    def __init__(self,llm_model):
        """
        llm_model: the Llama() instance from phase_two/llm.py
        """
        self.model=llm_model
    def classify(self, query:str)->TriageResults:
        """
        try a keyword classifier first (fast , deterministic)
        also run the model classification 
        if they aggres then it has a high confidence
        if the disagree then we go with the model 
        if the model output is unparseable then we use the keyword result
        """
        keyword_intent=_keyword_classify(query)
        model_intent=self._model_classify(query)

        if model_intent is None:
            return TriageResults(
                intent=keyword_intent,
                confidence="low",
                reasoning=f"keyword match {keyword_intent} (model output is unparseable)"
            )
        if model_intent == keyword_intent:
            confidence="high"
            reasoning=f"model and keywords agree ->{model_intent}"
        else:
            confidence="medium"
            reasoning=f"model->{model_intent} , keywords->{keyword_intent} (using model)"
        return TriageResults(
            intent=model_intent,
            confidence=confidence,
            reasoning=reasoning,
        )
    def _model_classify(self, query:str)->str| None:
        """
        Ask Phi-3 to classify the query.
        Returns the intent string, or None if output is unparseable.
        """

        prompt=CLASSIIFICATION_PROMPT.format(query=query)   
        try:
            #build the ChatML prompt directly
            full_prompt=(
                f"<|system|>\nYou are a classifier. Output ONE word only. <|end|> \n"
                f"<|user|>\n{prompt}<|end|>\n"
                f"<|assistant|>\n"
            )
            response=self._model._model._model(
                full_prompt,
                max_tokens=5,
                temperature=0.0,
                stop=["<|end|>", "\n", " ", "<|user|>"],
                echo=False
            )
            raw=response["choices"][0]["text"].strip().upper()
            clean =re.sub(r"[^A-Z]","",raw)
            if clean in INTENTS:
                return clean
            
            for intent in INTENTS:
                if intent.startswith(clean) or clean.startswith(intent[:4]):
                    return intent
                
            return None #unparseable
        except Exception:
            return None