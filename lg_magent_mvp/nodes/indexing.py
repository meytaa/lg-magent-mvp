from __future__ import annotations

from typing import TYPE_CHECKING, List, Dict, Any, Optional
import json
from pathlib import Path
import fitz  # PyMuPDF
from PIL import Image
from collections import defaultdict

from . import BaseNode

if TYPE_CHECKING:
    from ..app import AgentState

try:
    import google.generativeai as genai
    from google.generativeai.types import HarmCategory, HarmBlockThreshold
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False


def detect_page_elements(page, doc) -> List[Dict[str, Any]]:
    """
    Use PDF parser to detect all elements on the page (images, tables, text).
    Returns a list of detected elements with their types and basic info.
    """
    elements = []
    
    # Detect images
    detected_images = page.get_images(full=True)
    for i, img_info in enumerate(detected_images):
        bbox = page.get_image_bbox(img_info)
        elements.append({
            "type": "image",
            "bbox": [round(c) for c in bbox],
            "img_info": img_info,  # Store for later extraction
            "order": len(elements)  # Reading order
        })
    
    # Detect tables
    try:
        table_finder = page.find_tables()
        tables = table_finder.tables if hasattr(table_finder, 'tables') else []
        for i, table in enumerate(tables):
            bbox = table.bbox
            elements.append({
                "type": "table",
                "bbox": [round(c) for c in bbox],
                "table_data": table.extract(),  # Extract table data
                "order": len(elements)
            })
    except Exception as e:
        print(f"  - Warning: Table detection failed: {e}")
        # Continue without tables
    
    # Detect text blocks (excluding areas covered by images/tables)
    text_blocks = page.get_text("dict")["blocks"]
    for block in text_blocks:
        if "lines" in block:  # Text block
            bbox = block["bbox"]
            # Check if this text block overlaps significantly with images/tables
            overlaps = False
            for elem in elements:
                if elem["type"] in ["image", "table"]:
                    # Simple overlap check - you could make this more sophisticated
                    elem_bbox = elem["bbox"]
                    if (bbox[0] < elem_bbox[2] and bbox[2] > elem_bbox[0] and 
                        bbox[1] < elem_bbox[3] and bbox[3] > elem_bbox[1]):
                        overlaps = True
                        break
            
            if not overlaps:
                # Extract text from this block
                text_content = ""
                for line in block["lines"]:
                    for span in line["spans"]:
                        text_content += span["text"]
                    text_content += "\n"
                
                if text_content.strip():  # Only add non-empty text
                    elements.append({
                        "type": "text",
                        "content": text_content.strip(),
                        "bbox": [round(c) for c in bbox],
                        "order": len(elements)
                    })
    
    # Sort by reading order (top to bottom, left to right)
    elements.sort(key=lambda x: (x["bbox"][1], x["bbox"][0]))  # Sort by y, then x
    
    # Join sequential text blocks
    merged_elements = []
    i = 0
    while i < len(elements):
        current_element = elements[i]
        
        if current_element["type"] == "text":
            # Start collecting sequential text blocks
            combined_text = current_element["content"]
            combined_bbox = current_element["bbox"]
            j = i + 1
            
            # Look ahead for more sequential text blocks
            while j < len(elements) and elements[j]["type"] == "text":
                # Check if the next text block is reasonably close (sequential)
                current_bottom = combined_bbox[3]  # bottom y coordinate
                next_top = elements[j]["bbox"][1]   # top y coordinate
                vertical_gap = next_top - current_bottom
                
                # If the gap is reasonable (less than 50 pixels), merge them
                if vertical_gap < 50:
                    combined_text += "\n" + elements[j]["content"]
                    # Expand bounding box to include the new text
                    combined_bbox = [
                        min(combined_bbox[0], elements[j]["bbox"][0]),  # min x
                        min(combined_bbox[1], elements[j]["bbox"][1]),  # min y
                        max(combined_bbox[2], elements[j]["bbox"][2]),  # max x
                        max(combined_bbox[3], elements[j]["bbox"][3])   # max y
                    ]
                    j += 1
                else:
                    break  # Gap too large, stop merging
            
            # Add the merged text element
            merged_elements.append({
                "type": "text",
                "content": combined_text.strip(),
                "bbox": combined_bbox,
                "order": len(merged_elements)
            })
            
            i = j  # Skip all the merged elements
        else:
            # Non-text element, add as-is
            merged_elements.append(current_element)
            i += 1
    
    return merged_elements


def get_llm_semantic_analysis(page_image: Image.Image, detected_elements: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Sends detected elements to LLM for semantic analysis (descriptions, tags, captions).
    The LLM doesn't do classification - it only provides semantic insights.
    """
    if not GEMINI_AVAILABLE:
        print("  - LLM Semantic Analysis: Gemini not available, skipping")
        return {"elements": []}
        
    print(f"  - LLM Semantic Analysis: Analyzing {len(detected_elements)} detected elements")
    model = genai.GenerativeModel('gemini-1.5-flash')
    
    # Build element descriptions for the prompt
    element_descriptions = []
    for i, elem in enumerate(detected_elements):
        if elem['type'] == 'image':
            element_descriptions.append(f"Element {i+1}: IMAGE at position {elem.get('bbox', 'unknown')}")
        elif elem['type'] == 'table':
            element_descriptions.append(f"Element {i+1}: TABLE at position {elem.get('bbox', 'unknown')}")
        elif elem['type'] == 'text':
            text_preview = elem['content'][:100] + "..." if len(elem['content']) > 100 else elem['content']
            element_descriptions.append(f"Element {i+1}: TEXT - '{text_preview}'")
    
    elements_text = "\n".join(element_descriptions)
    
    prompt = f"""
    You are analyzing a document page image. I have already detected the following elements using a PDF parser:

    {elements_text}

    Your task is to provide semantic analysis for each element. Return a JSON object with this structure:
    {{
        "elements": [
            // For each element in order:
            {{
                "element_index": 0,  // 0-based index matching the input list
                "semantic_info": {{
                    // For images: provide "description" and "tags" 
                    // For tables: provide "caption" and "summary"
                    // For text: provide "summary" if it's long, otherwise null
                }}
            }}
        ]
    }}

    For images: Provide a detailed description of what you see and relevant tags.
    For tables: Provide a caption/title and brief summary of the table content.
    For text: Only provide a summary if the text is very long (>200 chars), otherwise leave semantic_info as null.

    Look at the page image and provide semantic analysis for the detected elements.
    """

    try:
        print(f"  - LLM Semantic Analysis: Making API call to Gemini...")
        response = model.generate_content([prompt, page_image])
        print(f"  - LLM Semantic Analysis: Received response (length: {len(response.text)} chars)")
        
        # Clean up the response to extract the pure JSON
        json_text = response.text.strip().replace("```json", "").replace("```", "")
        parsed_json = json.loads(json_text)
        
        return parsed_json
    except Exception as e:
        print(f"  - LLM Semantic Analysis: ERROR - {e}")
        return {"elements": []}


def process_pdf_hybrid(pdf_path: str, output_dir: str) -> Dict[str, Any]:
    """
    Process PDF using hybrid approach: PDF parser for structure + LLM for semantics.
    Returns structured content data.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories
    pdf_name = Path(pdf_path).stem
    doc_output_dir = output_path / pdf_name
    doc_output_dir.mkdir(exist_ok=True)
    img_output_dir = doc_output_dir / "images"
    img_output_dir.mkdir(exist_ok=True)
    
    doc = fitz.open(pdf_path)
    all_pages_data = []
    
    print(f"Processing {doc.page_count} pages...")
    
    for page_num in range(doc.page_count):
        page = doc.load_page(page_num)
        print(f"\n--- Processing Page {page_num + 1} ---")
        
        page_data = {
            "page": page_num + 1,
            "page_content": []
        }
        
        # Check if page is complex (has images or tables)
        has_images = len(page.get_images()) > 0
        try:
            table_finder = page.find_tables()
            has_tables = len(table_finder.tables) > 0 if hasattr(table_finder, 'tables') else False
        except:
            has_tables = False
        is_complex = has_images or has_tables
        
        print(f"  - Page complexity: {'Complex' if is_complex else 'Simple'} (images: {has_images}, tables: {has_tables})")
        
        if not is_complex:
            # Simple page: Extract text locally
            print("  - Extracting text locally...")
            text = page.get_text("text")
            page_data["page_content"] = [{"type": "text", "content": text}]
        else:
            print("  - Complex page: Using parser + LLM approach...")
            
            # Step 1: Use PDF parser to detect all elements
            print("  - Step 1: Detecting elements with PDF parser...")
            detected_elements = detect_page_elements(page, doc)
            print(f"  - Parser detected: {len(detected_elements)} elements")
            element_types = [elem['type'] for elem in detected_elements]
            print(f"  - Element types: {element_types}")
            
            # Step 2: Render page for LLM semantic analysis
            pix = page.get_pixmap(dpi=200)
            page_image = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
            
            # Step 3: Get semantic analysis from LLM
            print("  - Step 2: Getting semantic analysis from LLM...")
            semantic_analysis = get_llm_semantic_analysis(page_image, detected_elements)
            
            # Step 4: Build final JSON structure by combining parser results + LLM insights
            print("  - Step 3: Building final JSON structure...")
            final_content = []
            
            for i, element in enumerate(detected_elements):
                # Get semantic info from LLM if available
                semantic_info = {}
                if "elements" in semantic_analysis:
                    for sem_elem in semantic_analysis["elements"]:
                        if sem_elem.get("element_index") == i:
                            semantic_info = sem_elem.get("semantic_info", {})
                            break
                
                if element["type"] == "image":
                    # Save the image file
                    img_path = img_output_dir / f"page_{page_num+1}_img_{len([e for e in final_content if e['type'] == 'image']) + 1}.png"
                    img_xref = element["img_info"][0]
                    img_pix = fitz.Pixmap(doc, img_xref)
                    print(f"  - Saving image to {img_path}...")
                    img_pix.save(str(img_path))
                    
                    # Build image element
                    image_content = {
                        "description": semantic_info.get("description", "Image detected by PDF parser") if semantic_info else "Image detected by PDF parser",
                        "tags": semantic_info.get("tags", []) if semantic_info else [],
                        "caption": semantic_info.get("caption", "N/A") if semantic_info else "N/A",
                        "path": str(img_path),
                        "bbox": element["bbox"]
                    }
                    final_content.append({"type": "image", "content": image_content})
                    
                elif element["type"] == "table":
                    # Build table element
                    table_content = {
                        "caption": semantic_info.get("caption", "N/A") if semantic_info else "N/A",
                        "summary": semantic_info.get("summary", "Table detected by PDF parser") if semantic_info else "Table detected by PDF parser",
                        "data": element["table_data"],
                        "bbox": element["bbox"]
                    }
                    final_content.append({"type": "table", "content": table_content})
                    
                elif element["type"] == "text":
                    # Build text element
                    text_content = element["content"]
                    if semantic_info and semantic_info.get("summary"):
                        # If LLM provided a summary for long text, we could use it
                        # For now, just use the original text
                        pass
                    final_content.append({"type": "text", "content": text_content})
            
            page_data["page_content"] = final_content
        
        all_pages_data.append(page_data)
    
    doc.close()
    
    # Save the structured content
    content_output_path = output_path / f"{pdf_name}_content.json"
    with open(content_output_path, 'w', encoding='utf-8') as f:
        json.dump(all_pages_data, f, indent=2, ensure_ascii=False)
    
    print(f"\nHybrid processing complete!")
    print(f"Content saved to: {content_output_path}")
    
    return {
        "pages_processed": len(all_pages_data),
        "output_path": str(content_output_path),
        "content_data": all_pages_data
    }


class IndexingNode(BaseNode):
    """Node for hybrid PDF content indexing using parser + LLM approach."""

    def __init__(self):
        super().__init__(
            "indexing",
            description="Performs hybrid PDF content extraction: parser detects structure, LLM provides semantic analysis."
        )

    def execute(self, state: "AgentState") -> "AgentState":
        doc_path = self._get_doc_path(state)
        if not doc_path:
            return state

        # Create output directory
        output_dir = "output/indexing"
        
        try:
            # Process PDF with hybrid approach
            result = process_pdf_hybrid(doc_path, output_dir)
            
            # Store in context_data
            self._write_context_data(state, {
                "pages_processed": result["pages_processed"],
                "output_path": result["output_path"],
                "content_summary": {
                    "total_pages": result["pages_processed"],
                    "has_structured_content": True
                }
            })
            
            # Store detailed content in trace_data
            self._write_trace_data(state, result["content_data"], {
                "doc_path": doc_path,
                "indexing_method": "hybrid_parser_llm",
                "output_dir": output_dir
            })
            
            self._add_note(state, f"Indexed {result['pages_processed']} pages with hybrid approach")
            
        except Exception as e:
            self._add_note(state, f"Indexing failed: {str(e)}")
            self._write_context_data(state, {"indexing_failed": True, "error": str(e)})
        
        return state


# Create instance for backward compatibility
indexing_node = IndexingNode()
