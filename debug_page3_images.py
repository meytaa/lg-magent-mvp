import fitz  # PyMuPDF
from pathlib import Path

def debug_page3_images(pdf_path: str):
    """Debug what images are detected on page 3"""
    doc = fitz.open(pdf_path)
    page = doc[2]  # Page 3 (0-indexed)
    
    print(f"=== Page 3 Image Analysis ===")
    
    # Get all images
    images = page.get_images(full=True)
    print(f"Number of images detected: {len(images)}")
    
    for i, img in enumerate(images):
        print(f"\nImage {i+1}:")
        print(f"  - xref: {img[0]}")
        print(f"  - smask: {img[1]}")
        print(f"  - width: {img[2]}")
        print(f"  - height: {img[3]}")
        print(f"  - bpc: {img[4]}")
        print(f"  - colorspace: {img[5]}")
        print(f"  - alt: {img[6]}")
        print(f"  - name: {img[7]}")
        print(f"  - filter: {img[8]}")
        
        # Get image bbox
        bbox = page.get_image_bbox(img)
        print(f"  - bbox: {bbox}")
        
        # Try to extract the image
        try:
            img_xref = img[0]
            img_pix = fitz.Pixmap(doc, img_xref)
            print(f"  - Pixmap size: {img_pix.width}x{img_pix.height}")
            print(f"  - Pixmap colorspace: {img_pix.colorspace}")
            print(f"  - Pixmap samples length: {len(img_pix.samples)}")
            
            # Save for inspection
            output_path = f"page3_detected_image_{i+1}.png"
            img_pix.save(output_path)
            print(f"  - Saved to: {output_path}")

            # Also save to the proper output directory
            proper_output_path = "output/MM155 Deines Chiropractic-1/images/page_3_img_1.png"
            img_pix.save(proper_output_path)
            print(f"  - Also saved to proper location: {proper_output_path}")
            
        except Exception as e:
            print(f"  - Error extracting image: {e}")

if __name__ == "__main__":
    pdf_path = "/Users/mohammad/Documents/lg-magent-mvp/data/MM155 Deines Chiropractic-1.pdf"
    debug_page3_images(pdf_path)
