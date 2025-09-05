import fitz  # PyMuPDF
import openai
import os
import json
import argparse
from pathlib import Path
from PIL import Image
from dotenv import load_dotenv
from typing import Dict, Any, List, Optional, Union
from pydantic import BaseModel

# --- Response Schema Models ---
class ImageContent(BaseModel):
    id: str
    description: str
    caption: str

class TableContent(BaseModel):
    id: str
    description: str
    caption: str

class PageContentItem(BaseModel):
    type: str  # "text", "image", or "table"
    content: Union[str, ImageContent, TableContent]

class PageAnalysisResponse(BaseModel):
    page_content: List[PageContentItem]
    section: Optional[str]

# --- Configuration & Setup ---
load_dotenv()
try:
    openai.api_key = os.getenv("OPENAI_API_KEY")
    if not openai.api_key:
        raise ValueError("OPENAI_API_KEY not found")
except Exception as e:
    print(f"Error configuring OpenAI. Please ensure your OPENAI_API_KEY is set. Error: {e}")
    exit()

# --- Core Functions ---

def encode_image_to_base64(image: Image.Image) -> str:
    """Convert PIL Image to base64 string for OpenAI API."""
    import io
    import base64

    # Convert PIL Image to bytes
    buffer = io.BytesIO()
    image.save(buffer, format='PNG')
    buffer.seek(0)

    # Encode to base64
    image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    return image_base64

def get_llm_full_page_analysis(page_image: Image.Image, page_num: int, pdf_name: str) -> Dict[str, Any]:
    """
    Sends the full page image to LLM for complete analysis when images/tables are present.
    The LLM extracts all content and provides the same structure as the parser approach.
    """
    print(f"  - LLM Full Page Analysis: Analyzing complete page image")

    # Convert image to base64 for OpenAI API
    image_base64 = encode_image_to_base64(page_image)

    prompt = f"""
    You are analyzing a complete document page image (PAGE {page_num}). Your task is to extract all content and provide a structured analysis.

    Return a JSON object with this structure:
    {{
        "page_content": [
            // For each element found on the page in reading order:
            {{
                "type": "text|image|table",
                "content": {{
                    // For text: just the string content
                    // For images: {{"id": "fig_page{page_num}_Y", "description": "...", "caption": "..."}}
                    // For tables: {{"id": "table_page{page_num}_Y", "description": "...", "caption": "..."}}
                }}
            }}
        ],
        "section": "section name if identifiable, otherwise null"
    }}

    CRITICAL ID FORMATTING RULES:
    - This is PAGE {page_num} - ALL IDs must use page{page_num}
    - For images: "fig_page{page_num}_1", "fig_page{page_num}_2", etc. (increment Y for each image)
    - For tables: "table_page{page_num}_1", "table_page{page_num}_2", etc. (increment Y for each table)
    - NEVER use a different page number in the ID

    Instructions:
    - Extract ALL content from the page in proper reading order (top to bottom, left to right)
    - For text: Extract the actual text content as a string
    - For images: Provide description and caption (explicit or generated) - NO BBOX needed
    - For tables: Provide description and caption/title - NO BBOX needed
    - For section: Try to identify if this page belongs to a specific section (e.g., "Patient Information", "Treatment Plan", "Assessment", etc.)
    - Bounding boxes will be provided by the PDF parser, not by you
    - Maintain the exact same structure as the parser-based approach for consistency

    Analyze the complete page image and extract all content.
    """

    try:
        print(f"  - LLM Full Page Analysis: Making API call to OpenAI...")

        response = openai.beta.chat.completions.parse(
            model="gpt-4o-2024-08-06",  # Use GPT-4 with structured outputs
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{image_base64}"
                            }
                        }
                    ]
                }
            ],
            response_format=PageAnalysisResponse,
            max_tokens=4000
        )

        parsed_response = response.choices[0].message.parsed
        if not parsed_response:
            raise ValueError("Empty parsed response from OpenAI")

        print(f"  - LLM Full Page Analysis: Received structured response")
        print(f"  - LLM Response: {parsed_response.model_dump_json(indent=2)}")

        # Convert to dict for compatibility with existing code
        return parsed_response.model_dump()

    except Exception as e:
        print(f"  - LLM Full Page Analysis: ERROR - {e}")
        return {"page_content": [], "section": None}

def get_llm_semantic_analysis(page_image: Image.Image, detected_elements: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Sends detected elements to LLM for semantic analysis (descriptions, tags, captions).
    The LLM doesn't do classification - it only provides semantic insights.
    """
    print(f"  - LLM Semantic Analysis: Analyzing {len(detected_elements)} detected elements")

    # Convert image to base64 for OpenAI API
    image_base64 = encode_image_to_base64(page_image)

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
        print(f"  - LLM Semantic Analysis: Making API call to OpenAI...")

        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{image_base64}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=2000
        )

        response_text = response.choices[0].message.content
        if not response_text:
            raise ValueError("Empty response from OpenAI")

        print(f"  - LLM Semantic Analysis: Received response (length: {len(response_text)} chars)")

        # Clean up the response to extract the pure JSON
        json_text = response_text.strip().replace("```json", "").replace("```", "")
        parsed_json = json.loads(json_text)

        return parsed_json
    except Exception as e:
        print(f"  - LLM Semantic Analysis: ERROR - {e}")
        return {"elements": []}

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
    tables = page.find_tables()
    for i, table in enumerate(tables):
        bbox = table.bbox
        elements.append({
            "type": "table",
            "bbox": [round(c) for c in bbox],
            "table_data": table.extract(),  # Extract table data
            "order": len(elements)
        })

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

def process_pdf(pdf_path: str, output_dir: str):
    """
    Processes a PDF by analyzing each page. Simple text pages are extracted locally.
    Complex pages with images/tables are sent to a multimodal LLM for analysis.
    Results are cached.
    """
    pdf_name = Path(pdf_path).stem
    output_path = Path(output_dir) / f"{pdf_name}_content.json"
    img_output_dir = Path(output_dir) / pdf_name / "images"
    
    # --- Caching Logic ---
    if output_path.exists():
        print(f"✅ Cached result found at '{output_path}'. Skipping processing.")
        return

    img_output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Processing '{pdf_path}'. Outputs will be saved in '{output_dir}/{pdf_name}'.")
    
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
            pix = page.get_pixmap(dpi=200)
            page_image = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)

            # Save page image for debugging if it's page 3
            if page_num + 1 == 3:
                debug_path = f"debug_page3_full_image_v2.png"
                page_image.save(debug_path)
                print(f"  - DEBUG: Saved full page 3 image to {debug_path} for inspection")

            # Get full page analysis from LLM
            print("  - Getting full page analysis from LLM...")
            llm_analysis = get_llm_full_page_analysis(page_image, page_num + 1, pdf_name)

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
                    # For images, we need to extract and save them with real bbox
                    if detected_images and image_counter <= len(detected_images):
                        img_info = detected_images[image_counter - 1]
                        img_path = img_output_dir / f"page_{page_num+1}_img_{image_counter}.png"
                        img_xref = img_info[0]
                        img_pix = fitz.Pixmap(doc, img_xref)
                        print(f"  - Saving image to {img_path}...")
                        img_pix.save(str(img_path))

                        # Get real bbox from parser
                        real_bbox = page.get_image_bbox(img_info)

                        # Update the content with the actual file path and real bbox
                        if isinstance(content_item["content"], dict):
                            content_item["content"]["path"] = str(img_path)
                            content_item["content"]["bbox"] = [round(c) for c in real_bbox]
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

    # Create document summary
    document_summary = {
        "sections": document_sections,
        "total_pages": len(document_content),
        "total_tables": len(all_tables),
        "total_figures": len(all_figures),
        "tables": all_tables,
        "figures": all_figures
    }

    # Create final document structure with metadata
    final_document = {
        "document_meta": {
            "sections": document_sections,
            "total_pages": len(document_content)
        },
        "pages": document_content,
        "summary": document_summary
    }

    # Save the final JSON file, which serves as our cache
    with open(output_path, "w") as f:
        json.dump(final_document, f, indent=2)

    print(f"\n🎉 Successfully processed document. Output saved to '{output_path}'.")
    print(f"Found sections: {document_sections}")
    print(f"Found {len(all_tables)} tables and {len(all_figures)} figures")

# --- Main Execution Block ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Intelligently parse a PDF using local extraction for simple pages and an LLM for complex pages.")
    parser.add_argument("pdf_path", type=str, help="The full path to the PDF file.")
    parser.add_argument("--output_dir", type=str, default="output", help="Directory to save the JSON output and extracted images.")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.pdf_path):
        print(f"Error: The file '{args.pdf_path}' was not found.")
    else:
        process_pdf(args.pdf_path, args.output_dir)