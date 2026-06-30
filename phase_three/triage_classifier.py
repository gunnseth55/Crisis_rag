"""
phase3/triage_classifier.py — Intent classification before retrieval

CHANGE LOG (fix after observed misclassification):
    Original design trusted the model over keywords on disagreement.
    Observed failure: "my leg is bleeding and it won't stop" was
    classified EVACUATION by the model despite an unambiguous MEDICAL
    keyword match ("bleeding"). This is dangerous — it caused the
    MEDICAL-specific stricter retrieval threshold (0.50) to be skipped
    in favour of EVACUATION's looser threshold (0.45).

    FIX: keyword matches now take priority when they are unambiguous
    (score >= 2, or score == 1 for MEDICAL specifically, since a single
    strong medical keyword like "bleeding" or "unconscious" should never
    be overridden). The model is only used to break ties or classify
    queries with no keyword matches at all.

    This mirrors real-world triage protocols (e.g. START triage) where
    explicit symptom presence overrides general impression.
"""

import re
from dataclasses import dataclass


INTENTS = {"MEDICAL", "EVACUATION", "SURVIVAL", "EMOTIONAL", "GENERAL"}

# Keywords whose presence is considered an unambiguous medical signal —
# these should NEVER be overridden by the model, because a missed
# medical classification is the most dangerous failure mode.
STRONG_MEDICAL_SIGNALS = [
    "bleed", "unconscious", "not breath", "cpr", "chok",
    "heart attack", "seizure", "overdose", "poison",
]


@dataclass
class TriageResults:
    intent:     str
    confidence: str
    reasoning:  str


CLASSIFICATION_PROMPT = """Classify this emergency query into exactly one category.

MEDICAL:    injuries, bleeding, CPR, unconscious, not breathing, pain, wound, fracture, burn, heart attack, choking
EVACUATION: escape, leave, run, where to go, route, trapped, flood rising, get out, which way
SURVIVAL:   water purification, food safety, warmth, shelter, cold weather, fire, overnight, resources
EMOTIONAL:  scared, afraid, panic, hopeless, crying, alone, traumatized, cannot cope, worried, stressed
GENERAL:    what is, how does, explain, information, prepare, news, advice, general question

Query: "{query}"

Answer with ONE WORD only (MEDICAL / EVACUATION / SURVIVAL / EMOTIONAL / GENERAL):"""


KEYWORD_RULES = [
    ("MEDICAL",    ["bleed", "wound", "hurt", "injur", "pain",
                    "unconscious", "not breath", "cpr", "broken", "fractur",
                    "burn", "cut", "blood", "heart", "chok", "swallow",
                    "overdose", "poison", "breath", "pulse", "faint"]),
    ("EVACUATION", ["evacuat", "escape", "leave", "run", "flee",
                    "trapped", "stuck", "where to go", "route", "shelter",
                    "rising water", "get out", "exit", "way out", "flood"]),
    ("SURVIVAL",   ["water", "purif", "drink", "food", "eat", "warm", "cold",
                    "shelter", "fire", "overnight", "surviv", "heat", "freez",
                    "hypotherm", "frostbit", "dehydrat"]),
    ("EMOTIONAL",  ["scared", "afraid", "fear", "panic", "hopeless", "cry",
                    "alone", "traumat", "cannot cope", "worried", "anxiet",
                    "stress", "help me", "dont know what to do"]),
]


def _keyword_classify(query: str) -> tuple[str, int]:
    """
    Returns (intent, score) where score = number of matched keywords.
    Score is used to judge how confident the keyword match is.
    """
    q = query.lower()
    scores = {}
    for intent, keywords in KEYWORD_RULES:
        score = sum(1 for kw in keywords if kw in q)
        if score > 0:
            scores[intent] = score
    if not scores:
        return "GENERAL", 0
    best = max(scores, key=scores.get)
    return best, scores[best]


def _has_strong_medical_signal(query: str) -> bool:
    q = query.lower()
    return any(sig in q for sig in STRONG_MEDICAL_SIGNALS)


class TriageClassifier:
    """
    Classifies a crisis query into one of five intent categories.

    Priority order (changed after observed misclassification):
        1. Strong medical signal in keywords → MEDICAL, always, no override
        2. Keyword score >= 2 → trust keywords (model can't override)
        3. Keyword score == 1 and model agrees → high confidence
        4. Keyword score == 1 and model disagrees → trust keywords still
           (model is less reliable than a single clear keyword for safety)
        5. No keyword match at all → trust the model entirely
    """

    def __init__(self, llm_instance):
        self._model = llm_instance

    def classify(self, query: str) -> TriageResults:
        keyword_intent, keyword_score = _keyword_classify(query)
        model_intent = self._model_classify(query)

        # Rule 1: unambiguous medical signal always wins — safety first
        if _has_strong_medical_signal(query):
            return TriageResults(
                intent="MEDICAL",
                confidence="high",
                reasoning=f"strong medical keyword detected — overrides model ({model_intent})"
            )

        # Rule 2: strong keyword signal (2+ matches) — trust keywords
        if keyword_score >= 2:
            return TriageResults(
                intent=keyword_intent,
                confidence="high",
                reasoning=f"strong keyword match ({keyword_score} hits) → {keyword_intent}"
            )

        # Rule 3/4: weak keyword signal (1 match) — keywords still take priority
        if keyword_score == 1:
            if model_intent == keyword_intent:
                return TriageResults(
                    intent=keyword_intent, confidence="high",
                    reasoning=f"model and keyword agree → {keyword_intent}"
                )
            return TriageResults(
                intent=keyword_intent, confidence="medium",
                reasoning=f"keyword → {keyword_intent} (1 hit), model → {model_intent} (trusting keyword for safety)"
            )

        # Rule 5: no keyword signal — trust the model
        if model_intent is not None:
            return TriageResults(
                intent=model_intent, confidence="medium",
                reasoning=f"no keyword match — using model → {model_intent}"
            )

        return TriageResults(
            intent="GENERAL", confidence="low",
            reasoning="no keyword match and model output unparseable — defaulting to GENERAL"
        )

    def _model_classify(self, query: str) -> str | None:
        prompt = CLASSIFICATION_PROMPT.format(query=query)
        full_prompt = (
            f"<|system|>\nYou are a classifier. Output ONE word only.<|end|>\n"
            f"<|user|>\n{prompt}<|end|>\n"
            f"<|assistant|>\n"
        )
        try:
            response = self._model._model(
                full_prompt,
                max_tokens=5,
                temperature=0.0,
                stop=["<|end|>", "\n", " ", "<|user|>"],
                echo=False,
            )
            raw   = response["choices"][0]["text"].strip().upper()
            clean = re.sub(r"[^A-Z]", "", raw)

            if clean in INTENTS:
                return clean
            for intent in INTENTS:
                if intent.startswith(clean[:4]) or clean.startswith(intent[:4]):
                    return intent
            return None
        except Exception:
            return None