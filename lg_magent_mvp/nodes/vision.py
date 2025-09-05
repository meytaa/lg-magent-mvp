from typing import TYPE_CHECKING, List, Tuple, Dict, Any, Optional, Literal
import base64
import json
import os
from pathlib import Path
from PIL import Image
from pydantic import BaseModel

from ..schemas import Figure, make_figure_id
from ..models import get_vision_llm

from ..messages import create_figure_analysis_messages
from . import BaseNode

if TYPE_CHECKING:
    from ..app import AgentState


# --- Response Schema for Figure Analysis ---
class FigureAnalysisResult(BaseModel):
    figure_id: str
    analysis: str
    key_findings: List[str]
    relevance_to_question: str
    confidence: Literal["high", "medium", "low"]

class FigureAnalysisResponse(BaseModel):
    analyses: List[FigureAnalysisResult]
    overall_insights: str


# --- Cache Retrieval Functions ---
def _get_summary_cache_key(doc_path: str) -> str:
    """Generate cache key for summary data (same as in summary.py)."""
    import hashlib
    h = hashlib.sha256()
    h.update(doc_path.encode())
    h.update("hybrid_summary_v1".encode())
    try:
        st = os.stat(doc_path)
        h.update(str(st.st_size).encode())
        h.update(str(int(st.st_mtime)).encode())
    except Exception:
        pass
    return h.hexdigest()[:16]


def _load_figures_from_cache(doc_path: str, figure_ids: List[str]) -> List[Dict[str, Any]]:
    """Load specific figures from summary cache."""
    cache_dir = ".cache"
    key = _get_summary_cache_key(doc_path)
    summary_dir = os.path.join(cache_dir, "summary", key)
    content_path = os.path.join(summary_dir, "content.json")

    if not os.path.exists(content_path):
        print(f"⚠️ No summary cache found for {os.path.basename(doc_path)}")
        return []

    try:
        with open(content_path, "r") as f:
            cached_data = json.load(f)

        content_data = cached_data.get("content_data", {})
        pages = content_data.get("pages", [])

        # Extract figures matching the requested IDs
        found_figures = []
        for page_data in pages:
            for content_item in page_data.get("page_content", []):
                if content_item.get("type") == "image":
                    image_content = content_item.get("content", {})
                    if isinstance(image_content, dict):
                        fig_id = image_content.get("id", "")
                        if fig_id in figure_ids:
                            # Add page context
                            figure_data = image_content.copy()
                            figure_data["page"] = page_data.get("page", 1)
                            figure_data["section"] = page_data.get("section", "Unknown")
                            figure_data["page_text"] = page_data.get("integrated_text", "")
                            found_figures.append(figure_data)

        print(f"✅ Found {len(found_figures)} figures from cache: {[f.get('id') for f in found_figures]}")
        return found_figures

    except Exception as e:
        print(f"⚠️ Failed to load figures from cache: {e}")
        return []





def _nearest_caption(text_blocks: List[Tuple[List[float], str]], bbox: List[float]) -> str:
    x0, y0, x1, y1 = bbox
    candidates = []
    for tb, text in text_blocks:
        tx0, ty0, tx1, ty1 = tb
        overlap = max(0, min(x1, tx1) - max(x0, tx0))
        width = max(1, x1 - x0)
        if ty0 >= y1 and (ty0 - y1) < 50 and overlap / width > 0.4:
            if 5 <= len(text) <= 180:
                candidates.append(text)
    return candidates[0] if candidates else ""


def detect_figures(doc_path: str, max_per_page: int = 5) -> List[Figure]:
    try:
        import fitz  # PyMuPDF
    except Exception:
        return []

    out: List[Figure] = []
    doc = fitz.open(doc_path)
    for page_idx in range(doc.page_count):
        page = doc.load_page(page_idx)
        raw = page.get_text("rawdict")
        blocks = raw.get("blocks", []) if isinstance(raw, dict) else []
        img_blocks: List[Tuple[List[float], dict]] = []
        text_blocks: List[Tuple[List[float], str]] = []
        for b in blocks:
            btype = b.get("type")
            bbox = b.get("bbox")
            if btype == 1 and bbox:  # image
                img_blocks.append((bbox, b))
            elif btype == 0 and bbox:
                text = " ".join([s.get("text", "") for l in b.get("lines", []) for s in l.get("spans", [])])
                text_blocks.append((bbox, text.strip()))

        count = 0
        for idx, (bbox, _b) in enumerate(img_blocks, start=1):
            if count >= max_per_page:
                break
            caption = _nearest_caption(text_blocks, list(map(float, bbox)))
            out.append({
                "figure_id": make_figure_id(page_idx + 1, idx),
                "page": page_idx + 1,
                "caption": caption,
                "summary": "",  # filled after analysis
                "bbox": list(map(float, bbox)),
            })
            count += 1
    return out


def _encode_image_for_llm(page, bbox: List[float]) -> str:
    # Return a data URL (base64) for the cropped image region
    try:
        clip = fitz.Rect(*bbox)
        pix = page.get_pixmap(clip=clip, dpi=144)
        b = pix.tobytes("png")
        data_url = "data:image/png;base64," + base64.b64encode(b).decode("ascii")
        return data_url
    except Exception:
        return ""


class AnalyzeFiguresNode(BaseNode):
    """Node for analyzing specific figures from summary cache based on orchestrator input."""

    def __init__(self):
        super().__init__(
            "analyze_figures",
            description="Analyzes specific figures from summary cache to answer questions with detailed insights."
        )

    def execute(self, state: "AgentState") -> "AgentState":
        # Get orchestrator parameters
        next_action = state.get("next_action", {})
        parameters = next_action.get("parameters", {}) if next_action else {}

        # Extract figure IDs from orchestrator parameters
        figure_ids = parameters.get("figure_ids", [])
        if not figure_ids:
            self._add_note(state, "No figure IDs provided by orchestrator")
            self._add_agent_result(state, False, {"error": "No figure IDs specified"})
            return state

        # Get question and orchestrator thoughts for context
        question = state.get("question", "")
        orchestrator_history = state.get("orchestrator_chat_history", [])
        last_orchestrator_thoughts = ""
        if orchestrator_history:
            last_entry = orchestrator_history[-1]
            last_orchestrator_thoughts = last_entry.get("thoughts", "")

        # Get document path
        doc_path = self._get_doc_path(state)
        if not doc_path:
            self._add_note(state, "No document path available")
            self._add_agent_result(state, False, {"error": "No document path"})
            return state

        # Load figures from summary cache
        print(f"🔍 Loading figures from cache: {figure_ids}")
        figures = _load_figures_from_cache(doc_path, figure_ids)

        if not figures:
            self._add_note(state, f"No figures found in cache for IDs: {figure_ids}")
            self._add_agent_result(state, False, {
                "error": "Figures not found in cache",
                "requested_ids": figure_ids
            })
            return state

        try:
            print(f"🤖 Analyzing {len(figures)} figures with LLM...")

            # Get LLM and create structured output
            llm = get_vision_llm()
            messages = create_figure_analysis_messages(question, last_orchestrator_thoughts, figures, doc_path)

            # Use structured output to get response directly in the correct schema
            structured_llm = llm.with_structured_output(FigureAnalysisResponse)

            analysis_response = structured_llm.invoke(messages)

            print(f"✅ Received structured analysis response")

            # Convert Pydantic model to dict
            if hasattr(analysis_response, 'model_dump'):
                response_dict = analysis_response.model_dump()
            else:
                response_dict = analysis_response

            # Store results in context_data
            context_data = {
                "analyzed_figures": response_dict.get("analyses", []),
                "overall_insights": response_dict.get("overall_insights", ""),
                "figure_count": len(figures),
                "requested_ids": figure_ids,
                "question": question
            }
            self._write_context_data(state, context_data)

            # Store detailed trace data
            trace_data = {
                "figures_analyzed": figures,
                "analysis_results": response_dict,
                "orchestrator_thoughts": last_orchestrator_thoughts,
                "analysis_successful": True
            }
            self._write_trace_data(state, trace_data, {
                "doc_path": doc_path,
                "figure_ids": figure_ids,
                "analysis_method": "structured_llm_vision"
            })

            # Add structured result for orchestrator
            result_data = {
                "success": True,
                "figures_analyzed": len(figures),
                "analyses": response_dict.get("analyses", []),
                "overall_insights": response_dict.get("overall_insights", ""),
                "figure_ids": figure_ids
            }
            self._add_agent_result(state, True, result_data)

            # Create summary for notes
            analysis_summary = f"Analyzed {len(figures)} figures: "
            analysis_summary += ", ".join([f.get("id", "unknown") for f in figures])
            self._add_note(state, analysis_summary)

            # Clear next_action since we consumed it
            state["next_action"] = None

            return state

        except Exception as e:
            print(f"❌ Error in figure analysis: {e}")
            self._add_note(state, f"Error analyzing figures: {str(e)}")
            self._add_agent_result(state, False, {
                "error": str(e),
                "figure_ids": figure_ids
            })
            return state


# Create instance for backward compatibility
analyze_figures_node = AnalyzeFiguresNode()
