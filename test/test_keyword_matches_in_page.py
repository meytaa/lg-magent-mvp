"""
Function under test:

def keyword_matches_in_page(text: str, query: str) -> Iterable[Tuple[int, int]]:
    pattern = re.compile(re.escape(query), re.IGNORECASE)
    for m in pattern.finditer(text or ""):
        yield m.start(), m.end()

This uses re.escape for literal matching and re.IGNORECASE for case-insensitive
search over the provided text, yielding (start, end) byte offsets for each hit.
"""

import os
import pytest

from lg_magent_mvp.tools.retrievers import keyword_matches_in_page
from lg_magent_mvp.pdf import extract_text_pages


def test_single_match_span():
    text = "Hello world"
    query = "world"
    matches = list(keyword_matches_in_page(text, query))
    assert matches == [(6, 11)]


def test_case_insensitive_and_multiple_matches():
    text = "Hello World world"
    query = "world"
    matches = list(keyword_matches_in_page(text, query))
    # Expect matches for both "World" and "world"
    assert matches == [(6, 11), (12, 17)]


def test_special_characters_are_escaped():
    text = "Use C++ and regex (.*)"
    query = "C++"
    matches = list(keyword_matches_in_page(text, query))
    assert matches == [(4, 7)]


def test_none_text_yields_no_matches():
    # Function guards with (text or "") so None is safe
    matches = list(keyword_matches_in_page(None, "x"))  # type: ignore[arg-type]
    assert matches == []


def test_keyword_search_on_real_pdf():
    """Smoke test: read a real PDF and search for likely terms.

    Skips if PyMuPDF isn't installed, the file is missing, or the PDF has no
    extractable text (e.g., scanned image without OCR).
    """
    pdf_path = "data/Amaryllis_chiropractic_report.pdf"

    # Ensure dependency available
    try:
        import fitz  # noqa: F401
    except Exception:
        pytest.skip("PyMuPDF (fitz) not available")

    if not os.path.exists(pdf_path):
        pytest.skip(f"Missing test fixture PDF: {pdf_path}")

    pages = extract_text_pages(pdf_path)
    assert pages, "PDF should have at least one page"

    total_text_len = sum(len((p.get("text") or "")) for p in pages)
    if total_text_len == 0:
        pytest.skip("No extractable text; PDF may be scanned without OCR")

    # Try a few common, likely-to-exist clinical/report terms
    queries = [
        "chiropractic",
        "patient",
        "report",
        "date",
        "diagnosis",
        "assessment",
    ]

    total_hits = 0
    for page in pages:
        text = page.get("text") or ""
        for q in queries:
            hits = list(keyword_matches_in_page(text, q))
            # Validate match spans are within bounds
            for s, e in hits:
                assert 0 <= s < e <= len(text)
            total_hits += len(hits)

    assert total_hits >= 1, "Expected at least one keyword hit in the PDF"
