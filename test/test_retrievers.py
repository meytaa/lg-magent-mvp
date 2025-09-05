from lg_magent_mvp.tools.retrievers import (
    make_snippet,
    keyword_matches_in_page,
    chunk_text,
    build_chunks,
)


def test_make_snippet_and_matches():
    text = "Patient Name: John Doe. Diagnosis includes back pain and sciatica." * 2
    q = "sciatica"
    matches = list(keyword_matches_in_page(text, q))
    assert matches, "Should find at least one match"
    s, e = matches[0]
    snip = make_snippet(text, s, e, window=20)
    assert "sciatica".lower() in snip.lower()
    assert len(snip) <= 300


def test_chunk_text_and_build_chunks():
    page = {"page": 1, "text": "A" * 2000}
    spans = chunk_text(page["text"], size=800, overlap=150)
    assert spans[0] == (0, 800)
    assert spans[1][0] < spans[1][1] <= 2000
    chunks = build_chunks([page], size=800, overlap=150)
    # Expect ceiling((2000-800)/(800-150)) + 1 = 3 chunks
    assert len(chunks) == 3
    assert all(c.page == 1 for c in chunks)
    assert chunks[0].id.startswith("1-")
