import fitz  # PyMuPDF
import re
import os
import json
import argparse
from pathlib import Path
from PIL import Image
from transformers import pipeline
from typing import List, Dict, Any

# --- Configuration ---
# Create the classifier pipeline once and reuse it for efficiency.
# This downloads a model on first run (~500MB).
try:
    IMAGE_CLASSIFIER = pipeline(
    "zero-shot-image-classification",
    model="openai/clip-vit-large-patch14"
)
except Exception as e:
    print(f"Could not load the image classification model. The feature will be disabled. Error: {e}")
    IMAGE_CLASSIFIER = None

# --- Helper Functions ---

def get_document_structure(doc: fitz.Document) -> List[Dict[str, Any]]:
    """
    Analyzes the document to identify a structured table of contents
    based on font size and common heading patterns. This is more robust
    than a simple regex on the whole text.
    """
    structure = []
    for page in doc:
        blocks = page.get_text("dict", sort=True)["blocks"]
        for block in blocks:
            if "lines" in block:
                for line in block["lines"]:
                    for span in line["spans"]:
                        # Heuristic: Larger font sizes and bold text often indicate headers.
                        # This regex looks for patterns like "1.", "1.2.", "A.", etc.
                        is_heading = span['flags'] & 2**4  # Bold flag
                        match = re.match(r'^(\d+(\.\d+)*\s+|[A-Z]\.\s+)(.+)', span['text'])
                        
                        if match and (span['size'] > 11 or is_heading):
                             structure.append({
                                 "level": span['text'].count('.'),
                                 "title": span['text'].strip(),
                                 "page": page.number,
                                 "y_coord": span['bbox'][1] # Y-coordinate for sorting
                             })
    # Remove duplicates and sort
    unique_structure = [dict(t) for t in {tuple(d.items()) for d in structure}]
    unique_structure.sort(key=lambda x: (x['page'], x['y_coord']))
    
    return unique_structure

def find_closest_text_below(page: fitz.Page, bbox: fitz.Rect) -> str:
    """
    Finds the text block most likely to be a caption for a given bounding box.
    A caption is assumed to be the closest text block directly below the element.
    """
    x0, y0, x1, y1 = bbox
    caption_candidate = "No caption found."
    min_dist = float('inf')
    
    # A small horizontal tolerance to find captions that aren't perfectly aligned
    horizontal_tolerance = (x1 - x0) / 2 

    blocks = page.get_text("dict")["blocks"]
    for block in blocks:
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                s_x0, s_y0, s_x1, s_y1 = span["bbox"]
                # Check if the span is below and horizontally close to the element
                if s_y0 > y1 and s_x0 < x1 + horizontal_tolerance and s_x1 > x0 - horizontal_tolerance:
                    dist = s_y0 - y1  # Vertical distance
                    if dist < min_dist:
                        min_dist = dist
                        caption_candidate = span["text"].strip()

    # If the closest text is too far away, it's likely not a caption.
    if min_dist > 75:  # Threshold in points (72 points = 1 inch)
        return "No caption found."

    return caption_candidate

def classify_image_locally(image_path: str) -> str:
    """
    Uses a local zero-shot classification model to determine the content of an image.
    """
    if IMAGE_CLASSIFIER is None:
        return "Image classification disabled."
    try:
        candidate_labels = ["table", "image", "text"]
        results = IMAGE_CLASSIFIER(image_path, candidate_labels=candidate_labels)
        top_result = results[0]
        return f"{top_result['label']} (Score: {top_result['score']:.2f})"
    except Exception as e:
        return f"Could not classify image: {e}"

def process_pdf(pdf_path: str, output_dir: str):
    """
    Main orchestration function to process the PDF and generate catalogs.
    """
    print("Starting PDF processing...")
    doc = fitz.open(pdf_path)
    
    print("1. Identifying document structure...")
    structure = get_document_structure(doc)
    if not structure:
        print("Warning: Could not identify a clear document structure.")
    
    figures_catalog = []
    tables_catalog = []
    
    current_section = "N/A - Preamble"
    
    # Create a temporary directory for extracted images
    img_tmp_dir = Path(output_dir) / "temp_images"
    img_tmp_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"2. Processing {len(doc)} pages for figures and tables...")
    for page_num, page in enumerate(doc):
        # Update the current section based on what's on the page
        for item in structure:
            if item['page'] == page_num:
                current_section = item['title']
        
        # --- Process Images ---
        for img_index, img in enumerate(page.get_images(full=True)):
            xref = img[0]
            try:
                img_bbox = page.get_image_bbox(img)
                pix = fitz.Pixmap(doc, xref)
                
                # Save image temporarily for classification
                img_path = img_tmp_dir / f"p{page_num+1}_img{img_index}.png"
                pix.save(str(img_path))
                
                figures_catalog.append({
                    "page_number": page_num + 1,
                    "section": current_section,
                    "image_path": str(img_path),
                    "bounding_box": [round(c) for c in img_bbox],
                    "likely_caption": find_closest_text_below(page, img_bbox),
                    "image_tags": classify_image_locally(str(img_path))
                })
            except Exception as e:
                print(f"Warning: Could not process image on page {page_num+1}. Error: {e}")

        # --- Process Tables ---
        # PyMuPDF's table finding is robust and a good starting point.
        for table in page.find_tables():
            tables_catalog.append({
                "page_number": page_num + 1,
                "section": current_section,
                "bounding_box": [round(c) for c in table.bbox],
                "likely_caption": find_closest_text_below(page, table.bbox),
                "header": table.header.names,
                "data_preview": table.extract()[0:3] # Preview first 3 rows
            })
            
    print("3. Finalizing catalogs...")
    # Save the catalogs to JSON files
    with open(Path(output_dir) / "figures_catalog.json", "w") as f:
        json.dump(figures_catalog, f, indent=2)
        
    with open(Path(output_dir) / "tables_catalog.json", "w") as f:
        json.dump(tables_catalog, f, indent=2)
        
    print(f"\n✅ Processing complete. Catalogs saved in '{output_dir}' directory.")
    print(f"   - Found {len(figures_catalog)} figures.")
    print(f"   - Found {len(tables_catalog)} tables.")


# --- Main Execution Block ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate structured catalogs of figures and tables from a PDF.")
    parser.add_argument("pdf_path", type=str, help="The full path to the PDF file to process.")
    parser.add_argument("--output_dir", type=str, default="output", help="Directory to save the output JSON files and images.")
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    process_pdf(args.pdf_path, args.output_dir)