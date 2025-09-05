from typing import TYPE_CHECKING, List, Dict, Optional

from ..schemas import Table, make_table_id
from . import BaseNode

if TYPE_CHECKING:
    from ..app import AgentState


def _build_table_from_rows(page_idx: int, t_idx: int, rows: List[List[Optional[str]]]) -> Table:
    rows = rows or []
    # Guess header from first non-empty row; otherwise use generic headers
    header_row: List[str] = []
    data_rows: List[List[str]] = []

    for row in rows:
        cells = [str(c or "").strip() for c in row]
        if not header_row and any(cells):
            header_row = cells
            continue
        if any(cells):
            data_rows.append(cells)

    if not header_row:
        # create generic headers based on longest row
        width = max((len(r or []) for r in rows), default=0)
        header_row = [f"col_{i+1}" for i in range(width)]
        for row in rows:
            cells = [str(c or "").strip() for c in row]
            if any(cells):
                data_rows.append(cells)

    # Basic quality heuristics
    empty_cells = sum(1 for r in data_rows for c in r if not c)
    total_cells = sum(len(r) for r in data_rows) or 1
    ocr_risk = empty_cells / total_cells > 0.5
    quality = {"ocr_risk": ocr_risk, "issues": []}

    table: Table = {
        "table_id": make_table_id(page_idx, t_idx),
        "page": page_idx,
        "title": ", ".join([c for c in header_row if c]).strip() or "Table",
        "headers": header_row,
        "rows": data_rows,
        "quality": quality,
    }
    return table


def extract_tables_from_pdf(doc_path: str, max_per_page: int = 5) -> List[Table]:
    import fitz  # PyMuPDF

    tables: List[Table] = []
    doc = fitz.open(doc_path)
    for page_idx in range(doc.page_count):
        page = doc.load_page(page_idx)
        res = page.find_tables()
        for t_idx, table in enumerate(res.tables[:max_per_page], start=1):
            rows = []
            try:
                rows = table.extract()
            except Exception:
                rows = []
            tbl = _build_table_from_rows(page_idx + 1, t_idx, rows or [])
            tables.append(tbl)
    return tables


class ExtractTablesNode(BaseNode):
    """Node for extracting tables from PDF documents."""

    def __init__(self):
        super().__init__(
            "extract_tables",
            description="Extracts tables from the PDF and adds them as structured evidence."
        )

    def execute(self, state: "AgentState") -> "AgentState":
        # Clear next_action since we're consuming it (if it was meant for us)
        next_action = state.get("next_action", {})
        if next_action and next_action.get("agent") == "extract_tables":
            state["next_action"] = None

        doc_path = self._get_doc_path(state)
        if not doc_path:
            return state

        tables = extract_tables_from_pdf(doc_path)

        if tables:
            # Store in legacy format for backward compatibility
            state.setdefault("tables", [])
            state["tables"].extend(tables)
            self._add_evidence(state, "tables", tables)

            # Store in new context_data format
            self._write_context_data(state, {"tables": tables, "count": len(tables)})

            # Store detailed trace data
            self._write_trace_data(state, {
                "tables": tables,
                "count": len(tables),
                "extraction_successful": True
            }, {
                "doc_path": doc_path,
                "extraction_method": "extract_tables_from_pdf"
            })

            # Add structured result for orchestrator
            result_data = {
                "tables": tables,
                "count": len(tables),
                "extraction_successful": True
            }
            self._add_agent_result(state, True, result_data)

            self._add_note(state, f"tables: {len(tables)}")
        else:
            # Store empty result in context_data
            self._write_context_data(state, {"tables": [], "count": 0})

            # Store trace data for failed extraction
            self._write_trace_data(state, {
                "tables": [],
                "count": 0,
                "extraction_successful": False
            }, {
                "doc_path": doc_path,
                "extraction_method": "extract_tables_from_pdf"
            })

            # Add structured result for orchestrator (no tables found)
            result_data = {
                "tables": [],
                "count": 0,
                "extraction_successful": False
            }
            self._add_agent_result(state, True, result_data)

            self._add_note(state, "No tables found")
        return state


# Create instance for backward compatibility
extract_tables_node = ExtractTablesNode()
