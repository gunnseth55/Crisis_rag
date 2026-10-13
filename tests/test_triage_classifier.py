"""
tests/test_triage_classifier.py — coverage for phase_three/triage_classifier.py

Focus: the priority rules described in the module's own docstring
(strong medical signal > keyword score >= 2 > keyword score == 1 > model
fallback > GENERAL default), since this is the safety-critical routing
logic in the whole system. A real Phi-3 model is never loaded — a fake
LLM stub stands in for `llm_instance` so these tests run in <1s with no
GPU/CPU model weights required.

Run with:  pytest tests/test_triage_classifier.py -v
"""
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).parent.parent))

from phase_three.triage_classifier import (
    TriageClassifier,
    _keyword_classify,
    _has_strong_medical_signal,
)
from shared.query_normalizer import normalize_query


class _FakeInnerModel:
    """Stands in for llama_cpp.Llama — callable, returns a canned completion."""

    def __init__(self, reply: str):
        self._reply = reply

    def __call__(self, *args, **kwargs):
        return {"choices": [{"text": self._reply}]}


class FakeLLM:
    """Stands in for phase_two.llm.LLM — only `._model` is used by the classifier."""

    def __init__(self, model_reply: str = " GENERAL"):
        self._model = _FakeInnerModel(model_reply)


def make_classifier(model_reply: str = " GENERAL") -> TriageClassifier:
    return TriageClassifier(FakeLLM(model_reply))


# ---------- _keyword_classify ----------

def test_keyword_classify_no_match_returns_general_zero_score():
    intent, score = _keyword_classify("what is the capital of Australia")
    assert intent == "GENERAL"
    assert score == 0


def test_keyword_classify_weather_is_not_survival():
    intent, score = _keyword_classify("what is the weather today")
    assert intent == "GENERAL"


def test_keyword_classify_single_medical_keyword():
    intent, score = _keyword_classify("I have a small cut")
    assert intent == "MEDICAL"
    assert score >= 1


def test_keyword_classify_multiple_evacuation_keywords():
    intent, score = _keyword_classify("water is rising, I need to evacuate and escape now")
    assert intent == "EVACUATION"
    assert score >= 2


# ---------- _has_strong_medical_signal ----------

def test_strong_medical_signal_detects_bleeding():
    assert _has_strong_medical_signal("my leg is bleeding and won't stop")


def test_strong_medical_signal_detects_heart_attack():
    assert _has_strong_medical_signal("I think he is having a heart attack")


def test_strong_medical_signal_false_for_unrelated_query():
    assert not _has_strong_medical_signal("what's the safest evacuation route for a flood")


# ---------- TriageClassifier.classify — priority rules ----------

def test_strong_medical_signal_overrides_model_disagreement():
    # This is the exact regression case from the module's changelog:
    # a strong medical keyword must win even if the model says otherwise.
    clf = make_classifier(model_reply=" EVACUATION")
    result = clf.classify("my leg is bleeding and it won't stop")
    assert result.intent == "MEDICAL"
    assert result.confidence == "high"


def test_strong_keyword_score_trusts_keywords_over_model():
    clf = make_classifier(model_reply=" GENERAL")
    result = clf.classify("water is rising, I need to evacuate and escape now")
    assert result.intent == "EVACUATION"
    assert result.confidence == "high"


def test_weak_keyword_and_model_agree_is_high_confidence():
    clf = make_classifier(model_reply=" SURVIVAL")
    # "cold" alone is a single SURVIVAL keyword hit
    result = clf.classify("it is very cold outside")
    assert result.intent == "SURVIVAL"
    assert result.confidence == "high"


def test_weak_keyword_beats_disagreeing_model_for_safety():
    clf = make_classifier(model_reply=" GENERAL")
    result = clf.classify("it is very cold outside")
    assert result.intent == "SURVIVAL"
    assert result.confidence == "medium"


def test_no_keyword_match_trusts_model():
    clf = make_classifier(model_reply=" GENERAL")
    result = clf.classify("what is the capital of Australia")
    assert result.intent == "GENERAL"
    assert result.confidence == "medium"


def test_no_keyword_match_and_unparseable_model_defaults_general():
    # Empty model output used to fall through to whatever intent happened
    # to be first in Python's set iteration order (the EMOTIONAL bug from
    # the module's changelog). It must now default cleanly to GENERAL.
    clf = make_classifier(model_reply="")
    result = clf.classify("what is the capital of Australia")
    assert result.intent == "GENERAL"
    assert result.confidence == "low"


def test_normalize_query_does_not_hallucinate_hazard_words():
    normalized = normalize_query("what is the capital of France")
    assert "fracture" not in normalized


def test_emotional_keywords_route_to_emotional_when_no_medical_signal():
    clf = make_classifier(model_reply=" EMOTIONAL")
    result = clf.classify("I am so scared and alone right now")
    assert result.intent == "EMOTIONAL"


def test_medical_signal_still_wins_even_with_emotional_language():
    # Someone panicking about a bleeding injury should still be routed
    # MEDICAL, not EMOTIONAL — the physical danger takes priority.
    clf = make_classifier(model_reply=" EMOTIONAL")
    result = clf.classify("I'm so scared, I'm bleeding a lot and don't know what to do")
    assert result.intent == "MEDICAL"


def test_medical_stems_do_not_match_inside_unrelated_words():
    assert not _has_strong_medical_signal("discuss heartburn prevention")
    assert _keyword_classify("explain the execution process") == ("GENERAL", 0)


def test_normalizer_preserves_observed_hazard_typos():
    assert "landslide" in normalize_query("lamdlside safety")
    normalized = normalize_query("blooding")
    assert "bleeding" in normalized
    assert "flooding" in normalized
    assert make_classifier(" EVACUATION").classify("blooding").intent == "MEDICAL"


def test_strong_rules_skip_model_inference():
    clf = make_classifier()

    def unexpected_model_call(query):
        raise AssertionError("Strong keyword rules should not run model inference")

    clf._model_classify = unexpected_model_call
    assert clf.classify("my leg is bleeding").intent == "MEDICAL"
    assert clf.classify("evacuate and escape now").intent == "EVACUATION"
