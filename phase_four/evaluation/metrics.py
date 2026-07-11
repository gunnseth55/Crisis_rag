"""
evaluation/metrics.py — scoring functions for the Phase 4 comparison harness
 
We don't have hand-written gold answers (writing 20-30 of those and keeping them
in sync with the knowledge base is its own project), so these are *proxy* metrics
computed against `expected_keywords` / `expected_intent` / `should_refuse` in
test_cases.json:
 
- keyword_score:  fraction of expected keywords that appear in the answer.
                   A rough stand-in for "did the system actually give the right
                   information", not a substitute for human review.
- is_refusal:      did the system decline to answer / point to 112 instead of
                   fabricating something.
- intent_match:    did the reported intent match the expected one (only
                   meaningful for configs that report an intent at all).
- source_grounded: crude faithfulness proxy — does the retrieved source list
                   contain at least one chunk (i.e. the answer wasn't generated
                   with zero retrieved context, which almost always means the
                   "content" is coming from the model's own fallback text).
 
None of these replace manually reading a sample of transcripts before the
numbers go in the paper — treat them as triage, not ground truth.
"""
 
REFUSAL_SIGNALS = [
    "don't have",
    "do not have",
    "call 112",
    "please call 112",
    "contact your local disaster management",
    "please contact emergency services",
]
 
def keyword_score(answer:str, expected_keywords:list[str])->float:
    if not expected_keywords:
        return 1.0 #nothing to check against
    a=answer.lower()
    hits=sum(1 for kw in expected_keywords if kw.lower() in a)
    return round(hits/len(expected_keywords),3)
def is_refusal(answer:str)->bool:
    a=answer.lower()
    return any(sig in a for sig in REFUSAL_SIGNALS)

def refusal_is_correct(answer: str, should_refuse: bool) -> bool:
    """
    Clearer version: for should_refuse cases we WANT a refusal.
    For normal cases we count it correct as long as it's not a refusal
    (a false refusal on an answerable question is a failure mode too).
    """
    refused = is_refusal(answer)
    if should_refuse:
        return refused
    return not refused
def intent_match(actual_intent: str | None, expected_intent: str | None) -> str:
    """
    Returns "match", "mismatch", or "n/a" (when either side has no concept
    of intent — e.g. the baseline RAG pipeline, or an out-of-scope case with
    no expected intent).
    """
    if expected_intent is None or actual_intent is None:
        return "n/a"
    if actual_intent in ("NO_TRIAGE",):
        return "n/a"
    return "match" if actual_intent == expected_intent else "mismatch"
 
 
def source_grounded(sources: list) -> bool:
    return bool(sources)
 