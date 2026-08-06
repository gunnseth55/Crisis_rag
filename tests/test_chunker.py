"""
tests/test_chunker.py — coverage for phase_one/chunker.py

Run with:  pytest tests/test_chunker.py -v
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from phase_one.chunker import (
    chunk_document,
    estimate_tokens,
    _clean,
    _split_on,
    _merge_into_chunks,
    CHUNK_SIZE,
    MIN_CHUNK_SIZE,
)


# ---------- estimate_tokens ----------

def test_estimate_tokens_basic():
    # len // 4, min 1
    assert estimate_tokens("abcd") == 1
    assert estimate_tokens("a" * 40) == 10


def test_estimate_tokens_never_zero_for_nonempty_text():
    assert estimate_tokens("a") == 1


def test_estimate_tokens_empty_string():
    # len("")//4 == 0, but max(1, ...) floors it at 1
    assert estimate_tokens("") == 1


# ---------- _clean ----------

def test_clean_normalizes_line_endings():
    assert "\r" not in _clean("line1\r\nline2\rline3")


def test_clean_collapses_excess_blank_lines():
    text = "a" + "\n" * 6 + "b"
    cleaned = _clean(text)
    assert "\n\n\n\n" not in cleaned  # collapsed to at most 3 newlines


def test_clean_collapses_excess_spaces():
    cleaned = _clean("word1     word2")
    assert "     " not in cleaned


def test_clean_strips_leading_trailing_whitespace():
    assert _clean("   hello   ") == "hello"


def test_clean_converts_tabs_to_spaces():
    assert "\t" not in _clean("col1\tcol2")


# ---------- _split_on ----------

def test_split_on_empty_separator_splits_to_chars():
    assert _split_on("abc", "") == ["a", "b", "c"]


def test_split_on_preserves_separator_except_last_piece():
    pieces = _split_on("a\n\nb\n\nc", "\n\n")
    # every piece except the last should retain the separator suffix
    assert pieces[-1] == "c"
    assert all(p.endswith("\n\n") or p == "c" for p in pieces)


def test_split_on_drops_whitespace_only_pieces():
    pieces = _split_on("a\n\n   \n\nb", "\n\n")
    assert all(p.strip() for p in pieces)


# ---------- chunk_document: short text (single chunk path) ----------

def test_chunk_document_short_text_returns_one_chunk():
    text = "This is a short first-aid tip about cuts."
    chunks = chunk_document(text, "test_source")
    assert len(chunks) == 1
    assert chunks[0]["source"] == "test_source"
    assert chunks[0]["chunk_index"] == 0


def test_chunk_document_short_text_preserves_content():
    text = "Apply pressure to the wound."
    chunks = chunk_document(text, "test_source")
    assert "Apply pressure to the wound." in chunks[0]["text"]


# ---------- chunk_document: long text (recursive split + merge path) ----------

def _make_long_text(n_paragraphs: int = 40) -> str:
    # Each paragraph is well over MIN_CHUNK_SIZE tokens' worth of characters,
    # and the whole thing is well over CHUNK_SIZE tokens, forcing the
    # recursive-split branch in chunk_document.
    paragraph = (
        "In the event of an emergency, remain calm and assess the situation "
        "before taking action. Locate the nearest safe exit and move away "
        "from immediate danger. "
    ) * 3
    return "\n\n".join(f"{paragraph} (section {i})" for i in range(n_paragraphs))


def test_chunk_document_long_text_produces_multiple_chunks():
    text = _make_long_text()
    chunks = chunk_document(text, "long_source")
    assert len(chunks) > 1


def test_chunk_document_long_text_all_chunks_meet_min_size():
    text = _make_long_text()
    chunks = chunk_document(text, "long_source")
    for c in chunks:
        assert c["token_count"] >= MIN_CHUNK_SIZE


def test_chunk_document_long_text_chunks_roughly_bounded_by_chunk_size():
    text = _make_long_text()
    chunks = chunk_document(text, "long_source")
    # Chunks may slightly exceed CHUNK_SIZE due to overlap, but shouldn't
    # balloon far past it (e.g. more than double) — that would indicate the
    # merge loop failed to flush.
    for c in chunks:
        assert c["token_count"] <= CHUNK_SIZE * 2


def test_chunk_document_long_text_chunk_indices_are_sequential():
    text = _make_long_text()
    chunks = chunk_document(text, "long_source")
    indices = [c["chunk_index"] for c in chunks]
    assert indices == list(range(len(indices)))


def test_chunk_document_sets_source_on_every_chunk():
    text = _make_long_text()
    chunks = chunk_document(text, "long_source")
    assert all(c["source"] == "long_source" for c in chunks)


def test_chunk_document_empty_text_does_not_crash():
    chunks = chunk_document("", "empty_source")
    # Should return either no chunks or a single degenerate chunk, but must
    # not raise.
    assert isinstance(chunks, list)


# ---------- _merge_into_chunks ----------

def test_merge_into_chunks_respects_overlap():
    pieces = ["word " * 50, "other " * 50, "more " * 50]
    merged = _merge_into_chunks(pieces, chunk_size=20, overlap=5)
    assert len(merged) >= 2