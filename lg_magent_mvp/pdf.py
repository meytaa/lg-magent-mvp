from __future__ import annotations

from typing import List, Dict
import os


def extract_text_pages(doc_path: str) -> List[Dict]:
    """Extract plain text per page using PyMuPDF (fitz).

    Returns a list of {"page": int, "text": str}. Pages are 1-based.
    """
    import fitz  # PyMuPDF

    if not os.path.exists(doc_path):
        raise FileNotFoundError(doc_path)

    pages: List[Dict] = []
    doc = fitz.open(doc_path)
    for i in range(doc.page_count):
        page = doc.load_page(i)
        text = page.get_text("text") or ""
        pages.append({"page": i + 1, "text": text})
    doc.close()
    return pages


def extract_structured_text_pages(doc_path: str) -> List[Dict]:
    """Extract structured text with font and layout information.

    Returns a list of {"page": int, "text": str, "blocks": List[Dict], "fonts": Dict}.
    """
    import fitz

    if not os.path.exists(doc_path):
        raise FileNotFoundError(doc_path)

    pages: List[Dict] = []
    doc = fitz.open(doc_path)

    for i in range(doc.page_count):
        page = doc.load_page(i)

        # Get plain text
        text = page.get_text("text") or ""

        # Get structured data
        raw_dict = page.get_text("rawdict")
        blocks = raw_dict.get("blocks", []) if isinstance(raw_dict, dict) else []

        # Extract font information
        fonts = {}
        for block in blocks:
            if block.get("type") == 0:  # text block
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        font_name = span.get("font", "")
                        font_size = span.get("size", 0)
                        font_flags = span.get("flags", 0)

                        if font_name and font_name not in fonts:
                            fonts[font_name] = {
                                "size": font_size,
                                "flags": font_flags,
                                "is_bold": bool(font_flags & 16),
                                "is_italic": bool(font_flags & 2),
                            }

        pages.append({
            "page": i + 1,
            "text": text,
            "blocks": blocks,
            "fonts": fonts,
        })

    doc.close()
    return pages
