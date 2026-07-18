"""
shared/query_normalizer.py — lightweight, offline typo correction for
crisis queries, used before embedding and before keyword-based intent
matching.

WHY THIS EXISTS
   
    Root cause: embedding models are much less robust to a typo in the one
    word that carries the query's specific meaning (the hazard name) than
    to typos elsewhere in a sentence. Misspelling "landslide" was enough to
    lose the strong semantic match to landslide-specific documents.

WHY APPEND, NOT REPLACE
    A naive "replace the word with its closest vocabulary match" approach
    is itself dangerous: character-level similarity doesn't respect
    meaning. E.g. "blooding" (a plausible typo for "bleeding") is CLOSER,
    character-for-character, to "flooding" (0.875 similarity) than to
    "bleeding" (0.75) — so a replace-based corrector would silently turn a
    medical emergency into a flood query. Instead, this appends every
    plausible correction after the original word, so downstream logic
    (including the triage classifier's existing MEDICAL-keyword-priority
    override) still sees every reading and can pick the safety-critical
    one, rather than being committed to this module's single best guess.

WHAT THIS IS NOT
    A general-purpose spell checker, or a substitute for a real retrieval
    fix (typo-tolerant vector search, cross-encoder reranking, hybrid
    keyword+vector retrieval). It's a small, cheap, deterministic patch for
    one specific, demonstrated, safety-relevant failure mode: a misspelled
    hazard name. Extend HAZARD_VOCABULARY if new hazard types are added to
    the knowledge base.
"""

import re
import difflib

HAZARD_VOCABULARY = [
    "landslide", "earthquake", "flood", "flooding", "cyclone", "tsunami",
    "drought", "heatwave", "coldwave", "glof", "nuclear", "radiological",
    "evacuation", "evacuate", "bleeding", "wound", "burn", "choking",
    "fracture", "unconscious", "breathing", "cpr", "sanitation",
    "hypothermia", "dehydration", "shelter", "debris", "trapped",
]

_MIN_WORD_LEN = 5   # skip short/common words entirely — too many false positives below this
_MATCH_CUTOFF = 0.7  # difflib similarity ratio; tuned against real typos, see module tests


def normalize_query(query: str, max_candidates: int = 2) -> str:
    """
    Returns the query with fuzzy-matched hazard-vocabulary candidates
    appended (not substituted) after any word that looks like a typo of
    one. Safe to call on every query — words that are already correct, or
    that don't resemble any hazard term, are left untouched.
    """
    def _replace(match: re.Match) -> str:
        word = match.group(0)
        lower = word.lower()
        if len(word) < _MIN_WORD_LEN or lower in HAZARD_VOCABULARY:
            return word
        matches = difflib.get_close_matches(
            lower, HAZARD_VOCABULARY, n=max_candidates, cutoff=_MATCH_CUTOFF
        )
        if not matches:
            return word
        return word + " " + " ".join(matches)

    return re.sub(r"[A-Za-z']+", _replace, query)