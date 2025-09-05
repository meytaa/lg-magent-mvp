from __future__ import annotations

from typing import TypedDict, List, Dict, Optional, Literal, Any
from datetime import datetime


# ----- Enums / Literals -----
Severity = Literal["critical", "major", "minor", "info"]


# ----- Core TypedDicts -----
class Citation(TypedDict, total=False):
    page: int  # 1-based page index
    span: List[int]  # [start_offset, end_offset] in page text (optional)
    bbox: List[float]  # [x0, y0, x1, y1] PDF coords (optional)
    table_id: str
    figure_id: str


class Finding(TypedDict):
    id: str
    type: str  # e.g., metadata, completeness, coding, billing, clinical_quality, formatting
    severity: Severity
    confidence: int  # 0–100
    rationale: str
    citation: Citation
    evidence: str  # short snippet (<= 300 chars)
    remediation: str


class Table(TypedDict):
    table_id: str
    page: int
    title: Optional[str]
    headers: List[str]
    rows: List[List[str]]
    quality: Dict[str, Any]  # e.g., {"ocr_risk": bool, "issues": [str]}


class Figure(TypedDict, total=False):
    figure_id: str
    page: int
    caption: Optional[str]
    summary: str
    bbox: List[float]


class ReportMeta(TypedDict, total=False):
    timestamp: str
    file: str
    pages: int
    model_versions: Dict[str, str]


class DocumentMeta(TypedDict, total=False):
    patient_name: Optional[str]
    patient_id: Optional[str]
    date_of_birth: Optional[str]
    date_of_service: Optional[str]
    provider_name: Optional[str]
    provider_id: Optional[str]
    sections: List[str]


class Report(TypedDict):
    report_meta: ReportMeta
    document_meta: DocumentMeta
    findings: List[Finding]
    tables: List[Table]
    figures: List[Figure]
    metrics: Dict[str, Any]
    narrative: str


# ----- Helper functions -----
def make_table_id(page: int, index: int) -> str:
    return f"T{page}-{index}"


def make_figure_id(page: int, index: int) -> str:
    return f"F{page}-{index}"


def normalize_severity(s: str) -> Severity:
    s_norm = (s or "").strip().lower()
    if s_norm in {"critical", "blocker", "p0"}:
        return "critical"
    if s_norm in {"major", "high", "p1"}:
        return "major"
    if s_norm in {"minor", "medium", "low", "p2", "p3"}:
        return "minor"
    return "info"


def clamp_confidence(v: Any) -> int:
    try:
        val = int(round(float(v)))
    except Exception:
        val = 0
    return max(0, min(100, val))


def trim_evidence(text: str, limit: int = 300) -> str:
    if text is None:
        return ""
    t = str(text).strip()
    if len(t) <= limit:
        return t
    # Keep start and end context
    head = max(0, limit - 3)
    return t[:head] + "..."


def build_metrics(findings: List[Finding]) -> Dict[str, Any]:
    metrics: Dict[str, Any] = {
        "counts": {"critical": 0, "major": 0, "minor": 0, "info": 0},
        "by_type": {},
        "total_findings": 0,
    }
    for f in findings:
        sev = f.get("severity", "info")
        metrics["counts"][sev] = metrics["counts"].get(sev, 0) + 1
        t = f.get("type", "unknown")
        metrics["by_type"][t] = metrics["by_type"].get(t, 0) + 1
        metrics["total_findings"] += 1
    return metrics


def default_report(
    *,
    file: str = "",
    pages: int = 0,
    model_versions: Optional[Dict[str, str]] = None,
    document_meta: Optional[DocumentMeta] = None,
    findings: Optional[List[Finding]] = None,
    tables: Optional[List[Table]] = None,
    figures: Optional[List[Figure]] = None,
    narrative: str = "",
) -> Report:
    rep: Report = {
        "report_meta": {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "file": file,
            "pages": pages,
            "model_versions": model_versions or {},
        },
        "document_meta": document_meta or {},
        "findings": findings or [],
        "tables": tables or [],
        "figures": figures or [],
        "metrics": build_metrics(findings or []),
        "narrative": narrative or "",
    }
    return rep


# ----- Document preflight summary types -----
class TableOverview(TypedDict, total=False):
    table_id: str
    page: int
    header: str
    columns: List[str]
    rows_count: int


class FigureOverview(TypedDict, total=False):
    figure_id: str
    page: int
    caption: Optional[str]
    bbox: List[float]


class DocSummary(TypedDict, total=False):
    pages: int
    sections: List[str]
    counts: Dict[str, int]
    text_by_section: Dict[str, Dict[str, int]]  # Section -> {words: int, blocks: int}
    tables: List[TableOverview]
    figures: List[FigureOverview]


# ----- Orchestrator schemas -----
class AgentToCall(TypedDict):
    name: str  # Agent name to call
    input_arguments: Dict[str, Any]  # Input arguments for the agent


class OrchestratorResponse(TypedDict):
    thoughts: str  # Thoughts and reasoning
    agent_to_call: Optional[AgentToCall]  # Agent to call with input arguments, or None if done
    final_answer: Optional[str]  # Final answer when process is complete
    process_complete: Optional[bool]  # Signal that the entire process is done


class AgentResult(TypedDict):
    agent: str  # Which agent produced this result
    success: bool  # Whether agent succeeded
    data: Dict[str, Any]  # Structured result data
    error: Optional[str]  # Error message if failed
    metadata: Dict[str, Any]  # Additional metadata
