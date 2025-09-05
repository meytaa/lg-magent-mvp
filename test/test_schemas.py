from lg_magent_mvp.schemas import (
    make_table_id,
    make_figure_id,
    normalize_severity,
    clamp_confidence,
    trim_evidence,
    build_metrics,
    default_report,
)


def test_id_helpers():
    assert make_table_id(2, 1) == "T2-1"
    assert make_figure_id(5, 2) == "F5-2"


def test_severity_normalization():
    assert normalize_severity("CRITICAL") == "critical"
    assert normalize_severity("High") == "major"
    assert normalize_severity("medium") == "minor"
    assert normalize_severity("unknown") == "info"


def test_confidence_clamp():
    assert clamp_confidence(95.6) == 96
    assert clamp_confidence(-10) == 0
    assert clamp_confidence(200) == 100
    assert clamp_confidence("42") == 42


def test_trim_evidence():
    short = "A" * 50
    long = "B" * 500
    assert trim_evidence(short) == short
    out = trim_evidence(long, limit=100)
    assert len(out) <= 100
    assert out.endswith("...")


def test_build_metrics_and_default_report():
    findings = [
        {
            "id": "FIND-001",
            "type": "completeness",
            "severity": "major",
            "confidence": 88,
            "rationale": "Missing signature on page 1",
            "citation": {"page": 1},
            "evidence": "Signature block is empty",
            "remediation": "Collect provider signature",
        },
        {
            "id": "FIND-002",
            "type": "metadata",
            "severity": "minor",
            "confidence": 60,
            "rationale": "DOB appears only on page 4",
            "citation": {"page": 1},
            "evidence": "DOB not present in header",
            "remediation": "Add DOB to first page",
        },
    ]
    rep = default_report(file="data/MC15 Deines Chiropractic.pdf", pages=3, findings=findings, narrative="Summary")
    assert rep["report_meta"]["file"] == "data/MC15 Deines Chiropractic.pdf"
    assert rep["report_meta"]["pages"] == 3
    assert rep["metrics"]["total_findings"] == 2
    assert rep["metrics"]["counts"]["major"] == 1
    assert rep["metrics"]["counts"]["minor"] == 1
    assert rep["narrative"] == "Summary"

