from __future__ import annotations

from typing import TYPE_CHECKING, List, Dict, Tuple, Optional, Union, Literal
import re
import statistics
import os
import json
import base64
import io
import hashlib
from pathlib import Path
from collections import defaultdict, Counter
from PIL import Image
from pydantic import BaseModel

from ..pdf import extract_text_pages
from ..schemas import DocSummary, TableOverview, FigureOverview, make_table_id, make_figure_id
from ..config import load_settings
from ..models import get_vision_llm
from ..messages import create_summary_messages
from . import BaseNode

if TYPE_CHECKING:
    from ..app import AgentState


# --- Caching Utilities ---
def _summary_cache_key(doc_path: str) -> str:
    """Generate a unique cache key for a document summary based on path and metadata."""
    h = hashlib.sha256()
    h.update(doc_path.encode())
    # Add processing parameters to cache key
    h.update("hybrid_summary_v1".encode())
    try:
        # Include file size and mtime for invalidation
        st = os.stat(doc_path)
        h.update(str(st.st_size).encode())
        h.update(str(int(st.st_mtime)).encode())
    except Exception:
        pass
    return h.hexdigest()[:16]


def _summary_cache_dir(base_dir: str, key: str) -> str:
    """Get the directory path for a summary cache."""
    return os.path.join(base_dir, "summary", key)


def _load_cached_summary(doc_path: str) -> Optional[Dict]:
    """Load cached summary if it exists and is valid."""
    # Use .cache directory similar to faiss indexing
    cache_dir = ".cache"
    key = _summary_cache_key(doc_path)
    summary_dir = _summary_cache_dir(cache_dir, key)
    meta_path = os.path.join(summary_dir, "meta.json")
    content_path = os.path.join(summary_dir, "content.json")

    if not os.path.exists(meta_path) or not os.path.exists(content_path):
        return None

    try:
        # Validate cache metadata
        with open(meta_path, "r") as f:
            meta = json.load(f)

        # Check if cache is for the same document
        if meta.get("doc") != doc_path:
            return None

        # Load cached content
        with open(content_path, "r") as f:
            cached_content = json.load(f)

        print(f"✅ Using cached summary for {os.path.basename(doc_path)}")
        return cached_content

    except Exception as e:
        print(f"⚠️ Failed to load cached summary: {e}")
        return None


def _save_summary_cache(doc_path: str, summary_data: Dict) -> str:
    """Save summary data to cache."""
    # Use .cache directory similar to faiss indexing
    cache_dir = ".cache"
    key = _summary_cache_key(doc_path)
    summary_dir = _summary_cache_dir(cache_dir, key)

    os.makedirs(summary_dir, exist_ok=True)

    # Save metadata
    meta_path = os.path.join(summary_dir, "meta.json")
    with open(meta_path, "w") as f:
        json.dump({
            "doc": doc_path,
            "cache_key": key,
            "processing_method": "hybrid_summary",
            "version": 1,
            "pages": summary_data.get("pages_processed", 0),
            "sections": len(summary_data.get("sections", [])),
            "tables": len(summary_data.get("tables", [])),
            "figures": len(summary_data.get("figures", [])),
        }, f, indent=2)

    # Save content
    content_path = os.path.join(summary_dir, "content.json")
    with open(content_path, "w") as f:
        json.dump(summary_data, f, indent=2)

    print(f"💾 Cached summary for {os.path.basename(doc_path)} in {summary_dir}")
    return summary_dir


# --- Response Schema Models for LLM ---
class ImageContent(BaseModel):
    id: str
    description: str
    caption: str

class TableContent(BaseModel):
    id: str
    description: str
    caption: str

class PageContentItem(BaseModel):
    type: Literal["text", "image", "table"]  # Strict type validation
    content: Union[str, ImageContent, TableContent]

class PageAnalysisResponse(BaseModel):
    page_content: List[PageContentItem]
    section: Optional[str]


# --- Hybrid PDF Processing Functions ---
def encode_image_to_base64(image: Image.Image) -> str:
    """Convert PIL Image to base64 string for OpenAI API."""
    buffer = io.BytesIO()
    image.save(buffer, format='PNG')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    return image_base64



def get_llm_full_page_analysis(page_image: Image.Image, page_num: int) -> Dict:
    """Get structured analysis from LLM for complex pages with images/tables."""
    print(f"  - LLM Full Page Analysis: Analyzing page {page_num}")

    # Convert image to base64 for OpenAI API
    image_base64 = encode_image_to_base64(page_image)

    try:
        print(f"  - LLM Full Page Analysis: Making API call with structured output...")

        # Get LLM and create structured output
        llm = get_vision_llm()
        messages = create_summary_messages(page_num, image_base64)

        # Use structured output to get response directly in the correct schema
        structured_llm = llm.with_structured_output(PageAnalysisResponse)

        summary_response = structured_llm.invoke(messages)

        print(f"  - LLM Full Page Analysis: Received structured response")

        # Convert Pydantic model to dict
        if hasattr(summary_response, 'model_dump'):
            response_dict = summary_response.model_dump()
            print(f"  - LLM Response: {json.dumps(response_dict, indent=2)}")
            return response_dict
        else:
            # Already a dict
            print(f"  - LLM Response: {json.dumps(summary_response, indent=2)}")
            return summary_response

    except Exception as e:
        print(f"  - LLM Full Page Analysis: ERROR - {e}")
        return {"page_content": [], "section": None}


def process_pdf_hybrid(pdf_path: str, output_dir: str) -> Dict:
    """
    Process PDF using hybrid approach: simple text extraction for text-only pages,
    LLM analysis for complex pages with images/tables.
    """
    import fitz

    pdf_name = Path(pdf_path).stem

    print(f"Processing '{pdf_path}' with hybrid approach...")

    doc = fitz.open(pdf_path)
    document_content = []
    document_sections = []  # Track all sections found in the document

    for page_num, page in enumerate(doc):
        print(f"Processing page {page_num + 1}/{len(doc)}...")

        # --- Complexity Check ---
        has_images = bool(page.get_images())
        has_tables = bool(page.find_tables())
        is_complex = has_images or has_tables
        print(f"  - Images: {has_images}, Tables: {has_tables}, Complex: {is_complex}")

        page_data = {"page": page_num + 1, "section": None}

        if not is_complex:
            # Simple page: Extract text locally
            print("  - Extracting text locally...")
            text = page.get_text("text")
            page_data["page_content"] = [{"type": "text", "content": text}]
            page_data["integrated_text"] = text
        else:
            print("  - Complex page: Using full LLM analysis...")

            # Render page for LLM analysis
            pix = page.get_pixmap(dpi=80)
            page_image = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)

            # Get full page analysis from LLM
            print("  - Getting full page analysis from LLM...")
            llm_analysis = get_llm_full_page_analysis(page_image, page_num + 1)

            # Extract page content and section from LLM response
            page_content = llm_analysis.get("page_content", [])
            page_section = llm_analysis.get("section", None)

            # Get actual bounding boxes from parser
            detected_images = page.get_images(full=True)
            detected_tables = page.find_tables()

            # Process images to save them to files and add proper paths and real bboxes
            final_content = []
            image_counter = 1
            table_counter = 1
            text_parts = []  # Collect all text content for integration

            for content_item in page_content:
                if content_item["type"] == "image":
                    # For images, we just need to extract bbox (no file saving)
                    if detected_images and image_counter <= len(detected_images):
                        img_info = detected_images[image_counter - 1]

                        # Get real bbox from parser
                        real_bbox = page.get_image_bbox(img_info)

                        # Update the content with the real bbox (no file path needed)
                        if isinstance(content_item["content"], dict):
                            content_item["content"]["bbox"] = [round(c) for c in real_bbox]
                            content_item["content"]["page"] = page_num + 1
                        image_counter += 1
                elif content_item["type"] == "table":
                    # For tables, get real bbox from parser
                    if detected_tables and table_counter <= len(detected_tables):
                        table = detected_tables[table_counter - 1]
                        real_bbox = table.bbox

                        # Update with real bbox
                        if isinstance(content_item["content"], dict):
                            content_item["content"]["bbox"] = [round(c) for c in real_bbox]
                        table_counter += 1
                elif content_item["type"] == "text":
                    # Collect text content for integration
                    text_parts.append(content_item["content"])

                final_content.append(content_item)

            # Post-process: merge sequential text blocks
            merged_content = []
            i = 0
            while i < len(final_content):
                current_item = final_content[i]

                if current_item["type"] == "text":
                    # Start collecting sequential text blocks
                    combined_text = current_item["content"]
                    j = i + 1

                    # Look ahead for more sequential text blocks
                    while j < len(final_content) and final_content[j]["type"] == "text":
                        combined_text += "\n\n" + final_content[j]["content"]
                        j += 1

                    # Add the merged text element
                    merged_content.append({
                        "type": "text",
                        "content": combined_text
                    })

                    i = j  # Skip all the merged elements
                else:
                    # Non-text element, add as-is
                    merged_content.append(current_item)
                    i += 1

            # Create integrated text from all text content
            integrated_text = " ".join(text_parts)

            page_data["page_content"] = merged_content
            page_data["section"] = page_section
            page_data["integrated_text"] = integrated_text

            # Track sections found in the document
            if page_section and page_section not in document_sections:
                document_sections.append(page_section)

        document_content.append(page_data)

    # Generate summary of all tables and figures
    all_tables = []
    all_figures = []

    for page_data in document_content:
        for content_item in page_data.get("page_content", []):
            if content_item["type"] == "table" and isinstance(content_item["content"], dict):
                table_info = content_item["content"]
                all_tables.append({
                    "id": table_info.get("id", f"table_page{page_data['page']}_unknown"),
                    "caption": table_info.get("caption", "No caption available"),
                    "page": page_data["page"],
                    "section": page_data.get("section", "Unknown")
                })
            elif content_item["type"] == "image" and isinstance(content_item["content"], dict):
                image_info = content_item["content"]
                all_figures.append({
                    "id": image_info.get("id", f"fig_page{page_data['page']}_unknown"),
                    "caption": image_info.get("caption", image_info.get("description", "No caption available")),
                    "page": page_data["page"],
                    "section": page_data.get("section", "Unknown")
                })

    # Calculate text statistics (word counts)
    total_word_count = 0
    total_text_blocks = 0
    text_by_section = {}

    for page_data in document_content:
        page_word_count = 0
        page_text_blocks = 0
        section = page_data.get("section", "Unknown")

        # Count words from page_content items
        for content_item in page_data.get("page_content", []):
            if content_item.get("type") == "text":
                text_content = content_item.get("content", "")
                if isinstance(text_content, str):
                    # Count words by splitting on whitespace
                    words = text_content.split()
                    page_word_count += len(words)
                    page_text_blocks += 1

        # Also count words in integrated text
        integrated_text = page_data.get("integrated_text", "")
        if integrated_text:
            words = integrated_text.split()
            page_word_count += len(words)

        total_word_count += page_word_count
        total_text_blocks += page_text_blocks

        # Track text by section
        if section not in text_by_section:
            text_by_section[section] = {"words": 0, "blocks": 0}
        text_by_section[section]["words"] += page_word_count
        text_by_section[section]["blocks"] += page_text_blocks

    # Create document summary with text statistics
    document_summary = {
        "sections": document_sections,
        "total_pages": len(document_content),
        "total_tables": len(all_tables),
        "total_figures": len(all_figures),
        "total_word_count": total_word_count,
        "total_text_blocks": total_text_blocks,
        "text_by_section": text_by_section,
        "tables": all_tables,
        "figures": all_figures
    }

    # Create final document structure
    final_document = {
        "document_meta": {
            "sections": document_sections,
            "total_pages": len(document_content)
        },
        "pages": document_content,
        "summary": document_summary
    }

    doc.close()

    print(f"🎉 Successfully processed document with hybrid approach.")
    print(f"Found sections: {document_sections}")
    print(f"Found {len(all_tables)} tables and {len(all_figures)} figures")

    return {
        "pages_processed": len(document_content),
        "content_data": final_document,
        "sections": document_sections,
        "tables": all_tables,
        "figures": all_figures
    }


def _extract_font_info(doc_path: str) -> List[Dict]:
    """Extract detailed font and layout information from PDF."""
    import fitz

    font_data = []
    doc = fitz.open(doc_path)

    for page_idx in range(doc.page_count):
        page = doc.load_page(page_idx)
        raw_dict = page.get_text("rawdict")

        for block in raw_dict.get("blocks", []):
            if block.get("type") == 0:  # text block
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        text = span.get("text", "").strip()
                        if text:
                            font_data.append({
                                "page": page_idx + 1,
                                "text": text,
                                "font": span.get("font", ""),
                                "size": span.get("size", 0),
                                "flags": span.get("flags", 0),  # bold, italic, etc.
                                "bbox": span.get("bbox", []),
                                "line_bbox": line.get("bbox", []),
                            })

    doc.close()
    return font_data


def _analyze_font_patterns(font_data: List[Dict]) -> Dict:
    """Analyze font patterns to identify heading characteristics."""
    if not font_data:
        return {"sizes": [], "common_size": 12, "heading_threshold": 14}

    # Collect font sizes and their frequencies
    sizes = [item["size"] for item in font_data if item["size"] > 0]
    size_counts = Counter(sizes)

    # Find the most common (body text) size
    common_size = size_counts.most_common(1)[0][0] if size_counts else 12

    # Calculate heading threshold (larger than common size)
    unique_sizes = sorted(set(sizes))
    heading_threshold = common_size

    for size in unique_sizes:
        if size > common_size:
            heading_threshold = size
            break

    # Analyze font families
    fonts = [item["font"] for item in font_data if item["font"]]
    font_counts = Counter(fonts)
    common_fonts = [font for font, _ in font_counts.most_common(3)]

    return {
        "sizes": unique_sizes,
        "common_size": common_size,
        "heading_threshold": heading_threshold,
        "common_fonts": common_fonts,
        "size_distribution": dict(size_counts),
    }


def _is_heading_by_font(item: Dict, patterns: Dict) -> Tuple[bool, int]:
    """Determine if text is a heading based on font characteristics."""
    size = item["size"]
    flags = item["flags"]
    text = item["text"].strip()

    # Skip very short or very long text
    if len(text) < 3 or len(text) > 200:
        return False, 0

    # Skip text that looks like body content
    if text.endswith('.') and len(text) > 50:
        return False, 0

    confidence = 0

    # Font size analysis
    if size >= patterns["heading_threshold"]:
        confidence += 3
    elif size > patterns["common_size"]:
        confidence += 2

    # Bold text (flag 16)
    if flags & 16:
        confidence += 2

    # Italic text (flag 2) - less common for headings but possible
    if flags & 2:
        confidence += 1

    # Text pattern analysis
    text_lower = text.lower()

    # Numbered sections
    if re.match(r'^\d+\.?\s+[A-Za-z]', text):
        confidence += 2

    # Roman numerals
    if re.match(r'^[IVX]+\.?\s+[A-Za-z]', text):
        confidence += 2

    # All caps (but not too long)
    if text.isupper() and 5 <= len(text) <= 50:
        confidence += 2

    # Title case
    if text.istitle() and len(text.split()) <= 8:
        confidence += 1

    # Common heading words
    heading_words = ['introduction', 'conclusion', 'summary', 'abstract', 'background',
                    'methodology', 'results', 'discussion', 'references', 'appendix']
    if any(word in text_lower for word in heading_words):
        confidence += 1

    return confidence >= 3, confidence


def _detect_sections_by_font(doc_path: str, max_sections: int = 15) -> List[str]:
    """Detect document sections using font analysis."""
    font_data = _extract_font_info(doc_path)
    if not font_data:
        return []

    patterns = _analyze_font_patterns(font_data)

    sections = []
    seen = set()

    # Group by page and sort by position
    page_groups = defaultdict(list)
    for item in font_data:
        page_groups[item["page"]].append(item)

    for page_num in sorted(page_groups.keys()):
        page_items = sorted(page_groups[page_num],
                          key=lambda x: (x["line_bbox"][1] if x["line_bbox"] else 0))

        for item in page_items:
            is_heading, confidence = _is_heading_by_font(item, patterns)

            if is_heading:
                text = item["text"].strip()
                text_key = text.lower()

                if text_key not in seen:
                    seen.add(text_key)
                    sections.append(text)

                    if len(sections) >= max_sections:
                        return sections

    return sections


def _guess_sections(pages: List[Dict], max_sections: int = 12) -> List[str]:
    """Legacy section detection - kept as fallback."""
    sections: List[str] = []
    seen = set()
    heading_re = re.compile(r"^(\d+\.|[A-Z][A-Za-z ]{4,})$")
    for p in pages:
        for line in (p.get("text") or "").splitlines():
            line = re.sub(r"\s+", " ", line).strip()
            if not line or len(line) > 120:
                continue
            # all-caps or looks like a numbered/Capitalized heading
            if line.isupper() and len(line) >= 5:
                key = line.lower()
            elif heading_re.match(line):
                key = line.lower()
            else:
                continue
            if key not in seen:
                seen.add(key)
                sections.append(line)
            if len(sections) >= max_sections:
                return sections
    return sections


def _validate_table_structure(rows: List[List[str]]) -> Tuple[bool, float]:
    """Validate table structure and return confidence score."""
    if not rows or len(rows) < 3:  # Need at least 3 rows for a meaningful table
        return False, 0.0

    # Check for consistent column count
    col_counts = [len(row) for row in rows]
    if not col_counts:
        return False, 0.0

    most_common_cols = max(set(col_counts), key=col_counts.count)
    consistency = col_counts.count(most_common_cols) / len(col_counts)

    # Check for non-empty cells
    total_cells = sum(col_counts)
    non_empty_cells = sum(1 for row in rows for cell in row if str(cell).strip())
    fill_rate = non_empty_cells / total_cells if total_cells > 0 else 0

    # Check for meaningful content (not just whitespace/formatting)
    meaningful_rows = 0
    for row in rows:
        meaningful_cells = sum(1 for cell in row if str(cell).strip() and len(str(cell).strip()) > 1)
        if meaningful_cells >= 2:  # At least 2 meaningful cells per row
            meaningful_rows += 1

    meaningful_rate = meaningful_rows / len(rows) if rows else 0

    # More stringent validation
    is_valid = (
        consistency >= 0.8 and  # Higher consistency requirement
        fill_rate >= 0.4 and   # Higher fill rate requirement
        most_common_cols >= 3 and  # At least 3 columns
        meaningful_rate >= 0.5  # At least 50% of rows have meaningful content
    )

    confidence = (
        consistency * 0.4 +
        fill_rate * 0.3 +
        meaningful_rate * 0.2 +
        min(most_common_cols / 6, 1) * 0.1
    )

    return is_valid, confidence


def _extract_tables_pymupdf(doc_path: str, max_per_page: int = 3) -> List[Tuple[TableOverview, float]]:
    """Extract tables using PyMuPDF with confidence scoring."""
    import fitz

    tables_with_confidence = []
    doc = fitz.open(doc_path)

    for page_idx in range(doc.page_count):
        page = doc.load_page(page_idx)
        try:
            res = page.find_tables()
            for t_idx, table in enumerate(res.tables[:max_per_page], start=1):
                header_line = ""
                columns: List[str] = []
                rows = []

                try:
                    rows = table.extract()
                except Exception:
                    continue

                is_valid, confidence = _validate_table_structure(rows)
                if not is_valid:
                    continue

                if rows:
                    # Find best header row
                    for row in rows:
                        if any(cell for cell in row):
                            columns = [str(c or "").strip() for c in row]
                            header_line = ", ".join([c for c in columns if c])
                            break

                table_overview = {
                    "table_id": make_table_id(page_idx + 1, t_idx),
                    "page": page_idx + 1,
                    "header": header_line or "Table",
                    "columns": columns,
                    "rows_count": max(0, len(rows) - 1) if columns else len(rows),
                }

                tables_with_confidence.append((table_overview, confidence))

        except Exception:
            continue

    doc.close()
    return tables_with_confidence


def _extract_tables_pdfplumber(doc_path: str, max_per_page: int = 3) -> List[Tuple[TableOverview, float]]:
    """Extract tables using pdfplumber with multiple strategies."""
    try:
        import pdfplumber
    except ImportError:
        return []

    tables_with_confidence = []

    try:
        with pdfplumber.open(doc_path) as pdf:
            for page_idx, page in enumerate(pdf.pages):
                page_tables = []

                # Strategy 1: Default table extraction
                try:
                    tables = page.extract_tables()
                    for table_data in tables:
                        if table_data:
                            page_tables.append((table_data, "default"))
                except Exception:
                    pass

                # Strategy 2: Text-based extraction (better for borderless tables)
                try:
                    tables_text = page.extract_tables(table_settings={
                        "vertical_strategy": "text",
                        "horizontal_strategy": "text",
                        "snap_tolerance": 5,
                        "join_tolerance": 3,
                        "edge_min_length": 3,
                        "min_words_vertical": 3,
                        "min_words_horizontal": 1,
                    })
                    for table_data in tables_text:
                        if table_data:
                            page_tables.append((table_data, "text"))
                except Exception:
                    pass

                # Strategy 3: Lines-based extraction
                try:
                    tables_lines = page.extract_tables(table_settings={
                        "vertical_strategy": "lines",
                        "horizontal_strategy": "lines",
                        "snap_tolerance": 3,
                    })
                    for table_data in tables_lines:
                        if table_data:
                            page_tables.append((table_data, "lines"))
                except Exception:
                    pass

                # Process all found tables
                for t_idx, (table_data, strategy) in enumerate(page_tables[:max_per_page], start=1):
                    if not table_data:
                        continue

                    # Convert to list of lists of strings
                    rows = [[str(cell or "").strip() for cell in row] for row in table_data]

                    is_valid, confidence = _validate_table_structure(rows)
                    if not is_valid:
                        continue

                    # Extract header
                    header_line = ""
                    columns = []
                    if rows:
                        columns = rows[0]
                        header_line = ", ".join([c for c in columns if c])

                    table_overview = {
                        "table_id": make_table_id(page_idx + 1, t_idx),
                        "page": page_idx + 1,
                        "header": header_line or "Table",
                        "columns": columns,
                        "rows_count": max(0, len(rows) - 1) if columns else len(rows),
                    }

                    # Boost confidence based on strategy
                    if strategy == "text":
                        confidence *= 1.2  # Text strategy often better for borderless
                    elif strategy == "lines":
                        confidence *= 1.1
                    else:
                        confidence *= 1.05

                    tables_with_confidence.append((table_overview, confidence))

    except Exception:
        pass

    return tables_with_confidence


def _extract_tables_camelot(doc_path: str) -> List[Tuple[TableOverview, float]]:
    """Extract tables using Camelot (stream method for borderless tables)."""
    try:
        import camelot
    except ImportError:
        return []

    tables_with_confidence = []

    try:
        # Use stream method for borderless tables
        tables = camelot.read_pdf(doc_path, flavor='stream', pages='all')

        # Handle TableList iteration
        for i in range(len(tables)):
            table = tables[i]

            if table.df.empty:
                continue

            # Convert DataFrame to list of lists
            rows = table.df.values.tolist()
            # Add header row
            header_row = table.df.columns.tolist()
            if header_row and any(str(h).strip() for h in header_row):
                rows.insert(0, header_row)

            # Convert to strings
            rows = [[str(cell or "").strip() for cell in row] for row in rows]

            is_valid, base_confidence = _validate_table_structure(rows)
            if not is_valid:
                continue

            # Use Camelot's accuracy as confidence boost (Camelot is usually very accurate)
            camelot_confidence = table.accuracy / 100.0  # Convert percentage to decimal
            confidence = base_confidence * 0.5 + camelot_confidence * 0.5  # Give more weight to Camelot's accuracy

            # Extract header
            header_line = ""
            columns = []
            if rows:
                columns = rows[0]
                header_line = ", ".join([c for c in columns if c])

            # Determine page number (Camelot provides this)
            page_num = getattr(table, 'page', 1)

            table_overview = {
                "table_id": make_table_id(page_num, i + 1),
                "page": page_num,
                "header": header_line or f"Table on page {page_num}",
                "columns": columns,
                "rows_count": max(0, len(rows) - 1) if columns else len(rows),
            }

            tables_with_confidence.append((table_overview, confidence))

    except Exception as e:
        # For debugging
        print(f"Camelot extraction error: {e}")
        pass

    return tables_with_confidence


def _merge_table_results(pymupdf_tables: List[Tuple[TableOverview, float]],
                        pdfplumber_tables: List[Tuple[TableOverview, float]],
                        camelot_tables: Optional[List[Tuple[TableOverview, float]]] = None) -> List[TableOverview]:
    """Merge and deduplicate table results from different libraries."""

    # If Camelot found tables, prioritize them since they're usually very accurate
    if camelot_tables:
        camelot_pages = set(table[0]["page"] for table in camelot_tables)
        final_tables = []

        # Add all Camelot tables (they're high quality)
        for table, confidence in camelot_tables:
            final_tables.append(table)

        # Add non-Camelot tables only from pages where Camelot didn't find anything
        all_other_tables = []

        # Add PyMuPDF results
        for table, confidence in pymupdf_tables:
            if table["page"] not in camelot_pages:
                all_other_tables.append((table, confidence, "pymupdf"))

        # Add pdfplumber results
        for table, confidence in pdfplumber_tables:
            if table["page"] not in camelot_pages:
                all_other_tables.append((table, confidence, "pdfplumber"))

        # Process other tables by page
        page_tables = defaultdict(list)
        for table, confidence, source in all_other_tables:
            page_tables[table["page"]].append((table, confidence, source))

        for tables in page_tables.values():
            # Sort by confidence and take best result per page
            tables.sort(key=lambda x: x[1], reverse=True)
            if tables:
                final_tables.append(tables[0][0])  # Take the best one

        return final_tables

    else:
        # Fallback to original logic if no Camelot results
        all_tables = []

        # Add PyMuPDF results
        for table, confidence in pymupdf_tables:
            all_tables.append((table, confidence, "pymupdf"))

        # Add pdfplumber results
        for table, confidence in pdfplumber_tables:
            all_tables.append((table, confidence, "pdfplumber"))

        # Group by page and position, keep best result
        page_tables = defaultdict(list)
        for table, confidence, source in all_tables:
            page_tables[table["page"]].append((table, confidence, source))

        final_tables = []
        for tables in page_tables.values():
            # Sort by confidence and take best result per page
            tables.sort(key=lambda x: x[1], reverse=True)
            if tables:
                final_tables.append(tables[0][0])  # Take the best one per page

        return final_tables


def _scan_tables_overview(doc_path: str, max_per_page: int = 3) -> List[TableOverview]:
    """Enhanced table detection with multiple libraries and confidence scoring."""
    # Try PyMuPDF first (good for tables with clear borders)
    pymupdf_tables = _extract_tables_pymupdf(doc_path, max_per_page)

    # Try pdfplumber with multiple strategies (good for borderless tables)
    pdfplumber_tables = _extract_tables_pdfplumber(doc_path, max_per_page)

    # Try Camelot stream method (excellent for borderless tables)
    camelot_tables = _extract_tables_camelot(doc_path)

    # Merge results from all methods
    return _merge_table_results(pymupdf_tables, pdfplumber_tables, camelot_tables)


def _extract_figure_captions(text_blocks: List[Tuple[List[float], str]],
                           img_bbox: List[float]) -> Tuple[str, float]:
    """Enhanced caption detection with pattern matching and positioning."""
    x0, y0, x1, y1 = img_bbox
    candidates = []

    # Caption patterns to look for
    caption_patterns = [
        r'^(Figure|Fig\.?)\s*\d+[:\.\-\s]',
        r'^(Table|Tab\.?)\s*\d+[:\.\-\s]',
        r'^(Chart|Graph|Diagram)\s*\d*[:\.\-\s]',
        r'^\d+\.\s*[A-Z]',  # Numbered captions
    ]

    for tb, text in text_blocks:
        tx0, ty0, tx1, ty1 = tb
        text = text.strip()

        if not text or len(text) < 5 or len(text) > 300:
            continue

        # Calculate spatial relationships
        overlap_x = max(0, min(x1, tx1) - max(x0, tx0))
        overlap_y = max(0, min(y1, ty1) - max(y0, ty0))
        img_width = max(1, x1 - x0)
        img_height = max(1, y1 - y0)

        horizontal_overlap = overlap_x / img_width

        confidence = 0

        # Check for caption patterns
        for pattern in caption_patterns:
            if re.match(pattern, text, re.IGNORECASE):
                confidence += 3
                break

        # Position-based scoring
        # Below image (most common)
        if ty0 >= y1 and (ty0 - y1) < 80 and horizontal_overlap > 0.3:
            confidence += 2
        # Above image
        elif ty1 <= y0 and (y0 - ty1) < 60 and horizontal_overlap > 0.3:
            confidence += 1.5
        # Side by side (less common for captions)
        elif overlap_y > 0 and abs(tx0 - x1) < 100:
            confidence += 1

        # Text characteristics
        if text.endswith('.'):
            confidence += 0.5

        # Avoid very long paragraphs
        if len(text) > 150:
            confidence -= 1

        # Prefer shorter, more caption-like text
        if 10 <= len(text) <= 80:
            confidence += 1

        if confidence > 0:
            candidates.append((text, confidence))

    if candidates:
        # Return best candidate
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[0]

    return "", 0.0


def _filter_meaningful_images(img_blocks: List[Tuple[List[float], dict]]) -> List[Tuple[List[float], dict]]:
    """Filter out decorative images and keep meaningful figures."""
    meaningful_images = []

    for bbox, img_data in img_blocks:
        x0, y0, x1, y1 = bbox
        width = x1 - x0
        height = y1 - y0
        area = width * height

        # Filter criteria
        # Skip very small images (likely decorative)
        if width < 50 or height < 50 or area < 5000:
            continue

        # Skip very thin images (likely lines or borders)
        aspect_ratio = max(width, height) / min(width, height)
        if aspect_ratio > 20:
            continue

        meaningful_images.append((bbox, img_data))

    return meaningful_images


def _scan_figures_overview(doc_path: str, max_per_page: int = 5) -> List[FigureOverview]:
    """Enhanced figure detection with better caption matching."""
    figures_ov: List[FigureOverview] = []
    import fitz

    doc = fitz.open(doc_path)
    for page_idx in range(doc.page_count):
        page = doc.load_page(page_idx)
        raw = page.get_text("rawdict")
        blocks = raw.get("blocks", []) if isinstance(raw, dict) else []

        # Collect image blocks and text blocks
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

        # Filter meaningful images
        meaningful_images = _filter_meaningful_images(img_blocks)

        count = 0
        for idx, (bbox, _b) in enumerate(meaningful_images, start=1):
            if count >= max_per_page:
                break

            caption, confidence = _extract_figure_captions(text_blocks, bbox)

            # Only include figures with reasonable confidence or size
            x0, y0, x1, y1 = bbox
            area = (x1 - x0) * (y1 - y0)

            if confidence > 1.0 or area > 20000:  # Large images even without good captions
                figures_ov.append({
                    "figure_id": make_figure_id(page_idx + 1, idx),
                    "page": page_idx + 1,
                    "caption": caption if caption else f"Figure on page {page_idx + 1}",
                    "bbox": list(map(float, bbox)),
                })
                count += 1

    doc.close()
    return figures_ov


def _assess_detection_quality(doc_path: str, sections: List[str],
                            tables: List[TableOverview],
                            figures: List[FigureOverview]) -> Dict:
    """Assess the quality of detection results."""
    import fitz

    quality_metrics = {
        "section_detection": {
            "count": len(sections),
            "confidence": "medium",
            "issues": [],
        },
        "table_detection": {
            "count": len(tables),
            "confidence": "medium",
            "issues": [],
        },
        "figure_detection": {
            "count": len(figures),
            "confidence": "medium",
            "issues": [],
        },
        "overall_confidence": "medium",
    }

    # Assess section detection quality
    if len(sections) == 0:
        quality_metrics["section_detection"]["confidence"] = "low"
        quality_metrics["section_detection"]["issues"].append("No sections detected")
    elif len(sections) > 20:
        quality_metrics["section_detection"]["confidence"] = "low"
        quality_metrics["section_detection"]["issues"].append("Too many sections detected - may include false positives")
    elif 3 <= len(sections) <= 15:
        quality_metrics["section_detection"]["confidence"] = "high"

    # Assess table detection quality
    if tables:
        avg_rows = sum(t.get("rows_count", 0) for t in tables) / len(tables)
        if avg_rows < 2:
            quality_metrics["table_detection"]["confidence"] = "low"
            quality_metrics["table_detection"]["issues"].append("Tables have very few rows")
        elif avg_rows >= 5:
            quality_metrics["table_detection"]["confidence"] = "high"

    # Assess figure detection quality
    if figures:
        captioned_figures = sum(1 for f in figures
                              if f.get("caption") and len(str(f.get("caption", ""))) > 10)
        caption_rate = captioned_figures / len(figures)
        if caption_rate >= 0.7:
            quality_metrics["figure_detection"]["confidence"] = "high"
        elif caption_rate < 0.3:
            quality_metrics["figure_detection"]["confidence"] = "low"
            quality_metrics["figure_detection"]["issues"].append("Many figures lack proper captions")

    # Overall confidence
    confidences = [
        quality_metrics["section_detection"]["confidence"],
        quality_metrics["table_detection"]["confidence"],
        quality_metrics["figure_detection"]["confidence"],
    ]

    if all(c == "high" for c in confidences):
        quality_metrics["overall_confidence"] = "high"
    elif any(c == "low" for c in confidences):
        quality_metrics["overall_confidence"] = "low"

    return quality_metrics


def build_doc_summary_hybrid(doc_path: str) -> DocSummary:
    """Build comprehensive document summary using hybrid approach with caching."""
    try:
        # Check for cached summary first
        cached_result = _load_cached_summary(doc_path)
        if cached_result:
            # Convert cached result to legacy format
            content_data = cached_result.get("content_data", {})
            summary = content_data.get("summary", {})
            sections = summary.get("sections", [])
            tables = summary.get("tables", [])
            figures = summary.get("figures", [])
            total_pages = summary.get("total_pages", 0)

            # Convert to legacy format for backward compatibility
            legacy_tables = []
            for table in tables:
                legacy_tables.append({
                    "table_id": table.get("id", "unknown"),
                    "page": table.get("page", 1),
                    "header": table.get("caption", "Table"),
                    "columns": [],  # Not extracted in hybrid approach
                    "rows_count": 0,  # Not extracted in hybrid approach
                })

            legacy_figures = []
            for figure in figures:
                legacy_figures.append({
                    "figure_id": figure.get("id", "unknown"),
                    "page": figure.get("page", 1),
                    "caption": figure.get("caption", "Figure"),
                    "bbox": [0, 0, 0, 0],  # Real bbox is in the detailed content
                })

            # Get text statistics from cached summary
            total_word_count = summary.get("total_word_count", 0)
            total_text_blocks = summary.get("total_text_blocks", 0)
            text_by_section = summary.get("text_by_section", {})

            return {
                "pages": total_pages,
                "sections": sections,
                "counts": {
                    "tables": len(tables),
                    "figures": len(figures),
                    "pages": total_pages,
                    "text_words": total_word_count,
                    "text_blocks": total_text_blocks,
                },
                "text_by_section": text_by_section,
                "tables": legacy_tables,
                "figures": legacy_figures,
            }

        # No cache found, process with hybrid approach
        print(f"🔄 Processing {os.path.basename(doc_path)} with hybrid approach (no cache found)")
        output_dir = "output/summary"
        os.makedirs(output_dir, exist_ok=True)

        result = process_pdf_hybrid(doc_path, output_dir)

        # Cache the result
        _save_summary_cache(doc_path, result)

        content_data = result["content_data"]

        # Extract summary information
        summary = content_data.get("summary", {})
        sections = summary.get("sections", [])
        tables = summary.get("tables", [])
        figures = summary.get("figures", [])
        total_pages = summary.get("total_pages", 0)

        # Convert to legacy format for backward compatibility
        legacy_tables = []
        for table in tables:
            legacy_tables.append({
                "table_id": table.get("id", "unknown"),
                "page": table.get("page", 1),
                "header": table.get("caption", "Table"),
                "columns": [],  # Not extracted in hybrid approach
                "rows_count": 0,  # Not extracted in hybrid approach
            })

        legacy_figures = []
        for figure in figures:
            legacy_figures.append({
                "figure_id": figure.get("id", "unknown"),
                "page": figure.get("page", 1),
                "caption": figure.get("caption", "Figure"),
                "bbox": [0, 0, 0, 0],  # Real bbox is in the detailed content
            })

        # Get text statistics from new summary
        total_word_count = summary.get("total_word_count", 0)
        total_text_blocks = summary.get("total_text_blocks", 0)
        text_by_section = summary.get("text_by_section", {})

        return {
            "pages": total_pages,
            "sections": sections,
            "counts": {
                "tables": len(tables),
                "figures": len(figures),
                "pages": total_pages,
                "text_words": total_word_count,
                "text_blocks": total_text_blocks,
            },
            "text_by_section": text_by_section,
            "tables": legacy_tables,
            "figures": legacy_figures,
        }

    except Exception as e:
        print(f"Error in hybrid summary: {e}")
        # Fallback to basic text extraction
        pages = extract_text_pages(doc_path)
        return {
            "pages": len(pages),
            "sections": [],
            "counts": {
                "tables": 0,
                "figures": 0,
                "pages": len(pages),
            },
            "tables": [],
            "figures": [],
        }


class SummarizeDocNode(BaseNode):
    """Node for creating document summaries using hybrid PDF processing approach."""

    def __init__(self):
        super().__init__(
            "summarize",
            description="Builds a document summary using hybrid approach: simple text extraction for text-only pages, LLM analysis for complex pages with images/tables."
        )

    def execute(self, state: "AgentState") -> "AgentState":
        doc_path = self._get_doc_path(state)
        if not doc_path:
            return state

        try:
            # Use hybrid processing approach
            summary = build_doc_summary_hybrid(doc_path)

            # Store in legacy format for backward compatibility
            state["doc_summary"] = summary

            # Store in new context_data format
            self._write_context_data(state, summary)

            # Store detailed trace data
            self._write_trace_data(state, summary, {
                "doc_path": doc_path,
                "summary_method": "hybrid_processing"
            })

            # Legacy note for backward compatibility
            self._add_note(state,
                f"Doc summary (hybrid): pages={summary.get('pages',0)}, tables={summary['counts'].get('tables',0)}, figures={summary['counts'].get('figures',0)}"
            )

            return state

        except Exception as e:
            self._add_note(state, f"Error in hybrid summary processing: {str(e)}")
            # Fallback to basic processing
            try:
                pages = extract_text_pages(doc_path)
                summary = {
                    "pages": len(pages),
                    "sections": [],
                    "counts": {"tables": 0, "figures": 0, "pages": len(pages)},
                    "tables": [],
                    "figures": [],
                }
                state["doc_summary"] = summary
                self._write_context_data(state, summary)
                self._write_trace_data(state, {"error": str(e), "fallback_used": True}, {
                    "doc_path": doc_path,
                    "summary_method": "fallback_basic"
                })
            except Exception as fallback_error:
                self._add_note(state, f"Fallback processing also failed: {str(fallback_error)}")

            return state


# Create instance for backward compatibility
summarize_doc_node = SummarizeDocNode()
