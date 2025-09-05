import types

import pytest

import lg_magent_mvp.tools.retrievers as retrievers


def test_keyword_search_cross_page_snippet(monkeypatch):
    # Arrange: craft two pages where the match is at the very end of page 1,
    # so the snippet window will spill into page 2 when built from concatenated text.
    page1_text = ("Z" * 20) + "needle"
    page2_text = "C" * 50
    fake_pages = [
        {"page": 1, "text": page1_text},
        {"page": 2, "text": page2_text},
    ]

    # Monkeypatch extract_text_pages used within retrievers
    def fake_extract_text_pages(_doc_path: str):
        return fake_pages

    monkeypatch.setattr(retrievers, "extract_text_pages", fake_extract_text_pages)

    # Act
    hits = retrievers.keyword_search("needle", doc_path="dummy.pdf", max_hits_per_page=5)

    # Assert
    assert hits, "Expected at least one hit for 'needle'"
    h0 = hits[0]
    # Enclosing page should be 1 (match starts on page 1)
    assert h0["page"] == 1
    # Span should be page-local indices (20, 26)
    assert h0["span"] == [20, 26]
    # Snippet should include content from page 2 due to cross-page window
    snip = h0["snippet"]
    assert "needle" in snip
    assert "C" in snip, "Snippet should include next-page context"

