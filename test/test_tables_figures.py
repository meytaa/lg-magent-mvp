from lg_magent_mvp.nodes.tables import _build_table_from_rows
from lg_magent_mvp.nodes.vision import _nearest_caption


def test_build_table_from_rows_simple():
    rows = [
        ["Header A", "Header B", "Header C"],
        ["1", "2", "3"],
        ["4", "", "6"],
    ]
    t = _build_table_from_rows(2, 1, rows)
    assert t["table_id"] == "T2-1"
    assert t["headers"] == ["Header A", "Header B", "Header C"]
    assert len(t["rows"]) == 2
    assert t["quality"]["ocr_risk"] in {True, False}


def test_nearest_caption_overlap_logic():
    # Image bbox
    bbox = [100.0, 100.0, 300.0, 200.0]
    # Text blocks: one below within 50px and overlapping horizontally
    text_blocks = [
        ([90.0, 210.0, 310.0, 230.0], "Figure 1. Lumbar Spine X-ray"),
        ([0.0, 0.0, 50.0, 50.0], "Unrelated header"),
    ]
    cap = _nearest_caption(text_blocks, bbox)
    assert "Lumbar" in cap or cap.startswith("Figure 1")

