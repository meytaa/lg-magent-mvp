from lg_magent_mvp.nodes.summary import _guess_sections


def test_guess_sections_basic():
    pages = [
        {"page": 1, "text": "INTRODUCTION\nSome text here\nMETHODS\nResults"},
        {"page": 2, "text": "1. BACKGROUND\nData\nDISCUSSION"},
    ]
    sections = _guess_sections(pages)
    # Should include all-caps and numbered headings
    assert any(s.upper() == "INTRODUCTION" for s in sections)
    assert any("BACKGROUND" in s.upper() for s in sections)
    assert any(s.upper() == "METHODS" for s in sections)
