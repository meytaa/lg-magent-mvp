# pdf_summarizer.py
import re
import os
import json
import math
from pathlib import Path
from collections import Counter, defaultdict

import fitz  # PyMuPDF
import pdfplumber
import pandas as pd

try:
    import camelot
    HAS_CAMELOT = True
except Exception:
    HAS_CAMELOT = False


CAPTION_PATTERNS = [
    re.compile(r"^\s*(Figure|Fig\.?|Image)\s*[\dA-Za-z.\-:() ]{0,10}[:.\-–]\s*(.+)$", re.I),
    re.compile(r"^\s*(Table)\s*[\dA-Za-z.\-:() ]{0,10}[:.\-–]\s*(.+)$", re.I),
]

def _norm(text):
    return re.sub(r"\s+", " ", (text or "").strip())

def quantile(values, q):
    if not values: return None
    xs = sorted(values)
    pos = (len(xs)-1)*q
    floor, ceil = math.floor(pos), math.ceil(pos)
    if floor == ceil: return xs[int(pos)]
    lo, hi = xs[floor], xs[ceil]
    return lo + (hi-lo)*(pos-floor)

def bbox_center(b):
    x0,y0,x1,y1 = b
    return (0.5*(x0+x1), 0.5*(y0+y1))

def overlap_ratio(a0, a1, b0, b1):
    inter = max(0, min(a1, b1) - max(a0, b0))
    denom = max(a1-a0, b1-b0, 1e-6)
    return inter / denom

def lines_from_pymupdf_page(page):
    """Return list of dicts: {'text','bbox','size','bold','page'} and image blocks {'type':'image','bbox','page'}."""
    data = page.get_text("dict")
    lines = []
    images = []
    for b in data.get("blocks", []):
        if b.get("type") == 0:  # text
            for l in b.get("lines", []):
                spans = l.get("spans", [])
                if not spans: continue
                text = _norm("".join(s.get("text","") for s in spans))
                if not text: continue
                sizes = [s.get("size", 0) for s in spans]
                flags = [s.get("flags", 0) for s in spans]
                avg_size = sum(sizes)/len(sizes)
                boldish = any((f & 2) for f in flags)  # 2 -> bold
                bbox = l.get("bbox")
                lines.append({"text": text, "bbox": bbox, "size": avg_size, "bold": boldish})
        elif b.get("type") == 1:  # image
            images.append({"bbox": b.get("bbox")})
    return lines, images

def detect_heading_thresholds(all_sizes):
    """Return size thresholds + rank mapping for heading levels using quantiles + top unique sizes."""
    if not all_sizes:
        return {"thresh": None, "levels": []}
    # Heuristic: top 10-15% sizes are headings; also capture top 3 unique sizes
    q85 = quantile(all_sizes, 0.85)
    uniq = sorted(set(round(s,1) for s in all_sizes), reverse=True)
    top_sizes = uniq[:3]
    return {"thresh": q85, "levels": top_sizes}

def heading_level_for_size(size, levels):
    # Map size to level based on nearest of top_sizes
    if not levels: return None
    diffs = [(abs(size - s), i) for i, s in enumerate(levels)]
    _, idx = min(diffs)
    return idx + 1  # 1..3

def find_caption_near_bbox(lines, bbox, direction="below", x_overlap_min=0.3, dy_limit_pts=120, merge_next=True):
    """
    Find a caption line near bbox. Prefer lines matching caption regexes.
    - direction: 'below', 'above', or 'either'
    """
    x0,y0,x1,y1 = bbox
    candidates = []
    for ln in lines:
        tx = ln["text"]
        if not tx: continue
        lx0,ly0,lx1,ly1 = ln["bbox"]
        x_ov = overlap_ratio(x0, x1, lx0, lx1)
        if x_ov < x_overlap_min:
            continue
        vertical_ok = False
        if direction in ("below", "either") and (ly0 >= y1) and (ly0 - y1 <= dy_limit_pts):
            vertical_ok = True
        if direction in ("above", "either") and (y0 >= ly1) and (y0 - ly1 <= dy_limit_pts):
            vertical_ok = True
        if not vertical_ok: continue

        score = abs((ly0+ly1)/2 - (y0+y1)/2)  # distance
        # Caption regex bonus
        regex_bonus = -1000 if any(p.search(tx) for p in CAPTION_PATTERNS) else 0
        candidates.append((score + regex_bonus, ln))

    if not candidates:
        return None

    candidates.sort(key=lambda t: t[0])
    best = candidates[0][1]

    if not merge_next:
        return best["text"]

    # greedily concatenate following line(s) if they look like continuation (close below, similar left)
    best_text = best["text"]
    bx0, by0, bx1, by1 = best["bbox"]
    cursor_y = by1
    for ln in sorted(lines, key=lambda z: (z["bbox"][1], z["bbox"][0])):
        lx0, ly0, lx1, ly1 = ln["bbox"]
        if ly0 <= cursor_y:  # only next lines below
            continue
        # near-same left edge & within small gap
        if abs(lx0 - bx0) < 20 and 0 < (ly0 - cursor_y) < 18 and overlap_ratio(bx0, bx1, lx0, lx1) > 0.2:
            best_text += " " + ln["text"]
            cursor_y = ly1
        # stop if big gap
        if ly0 - cursor_y > 24:
            break

    return best_text

def export_image_region(page, bbox, out_path, dpi=200):
    r = fitz.Rect(bbox)
    # scale so that roughly matches dpi
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)
    pm = page.get_pixmap(matrix=mat, clip=r, alpha=False)
    pm.save(out_path)
    return out_path

def extract_tables_with_camelot(pdf_path, pages_str):
    # Try both flavors, merge unique results per page
    results = defaultdict(list)
    try:
        t1 = camelot.read_pdf(pdf_path, pages=pages_str, flavor="lattice", suppress_stdout=True)
    except Exception:
        t1 = []
    try:
        t2 = camelot.read_pdf(pdf_path, pages=pages_str, flavor="stream", suppress_stdout=True)
    except Exception:
        t2 = []
    for t in list(t1) + list(t2):
        p = int(t.parsing_report["page"])
        # bbox is (x1,y1,x2,y2) in PDF points; camelot stores in ._bbox in many versions
        bbox = getattr(t, "_bbox", None)
        results[p].append((t, bbox))
    return results  # page -> list of (table, bbox)

def extract_tables_with_pdfplumber(pdf_path):
    out = defaultdict(list)
    with pdfplumber.open(pdf_path) as pl:
        for pidx, page in enumerate(pl.pages, start=1):
            try:
                # table_settings can be tuned; here using default detection
                tables = page.find_tables()
            except Exception:
                tables = []
            for tb in tables:
                # Convert to DataFrame
                df = pd.DataFrame(tb.extract())
                bbox = tb.bbox  # (x0, top, x1, bottom)
                out[pidx].append((df, bbox))
    return out

def summarize_pdf(pdf_path, outdir="pdf_summary_out"):
    os.makedirs(outdir, exist_ok=True)
    doc = fitz.open(pdf_path)

    # Pass 1: collect all lines & images for heading stats
    pages_data = []
    all_sizes = []
    with pdfplumber.open(pdf_path) as _pl:  # we reuse for tables later
        pass
    for i, page in enumerate(doc, start=1):
        lines, images = lines_from_pymupdf_page(page)
        for ln in lines:
            all_sizes.append(ln["size"])
        pages_data.append({"page": i, "lines": lines, "images": images})

    head_cfg = detect_heading_thresholds(all_sizes)

    # Headings
    headings = []
    for p in pages_data:
        page_num = p["page"]
        for ln in p["lines"]:
            if head_cfg["thresh"] is None:
                continue
            if ln["size"] >= head_cfg["thresh"] and len(ln["text"]) <= 140:
                lvl = heading_level_for_size(ln["size"], head_cfg["levels"])
                if lvl is None: 
                    continue
                # Avoid capturing running headers/footers: ignore very top/bottom 5% of page
                x0,y0,x1,y1 = ln["bbox"]
                _, h = doc[page_num-1].rect.br
                if y0 < 0.05*h or y1 > 0.95*h:
                    continue
                headings.append({
                    "page": page_num,
                    "level": lvl,
                    "text": ln["text"],
                    "bbox": ln["bbox"],
                    "size": round(ln["size"],1)
                })

    # Tables
    tables = []
    # First try Camelot (if available)
    page_range = f"1-{len(doc)}"
    came = extract_tables_with_camelot(pdf_path, page_range) if HAS_CAMELOT else {}
    # Fallback/complement with pdfplumber
    pl_tables = extract_tables_with_pdfplumber(pdf_path)

    def nearest_caption_for_bbox(page_lines, bbox, label="Table"):
        # Prefer above or below; many styles put caption above tables.
        cap = find_caption_near_bbox(page_lines, bbox, direction="above")
        if cap and re.search(rf"^\s*{label}\b", cap, re.I): return cap
        cap2 = find_caption_near_bbox(page_lines, bbox, direction="below")
        return cap or cap2

    # From Camelot
    for pnum, items in came.items():
        page = doc[pnum-1]
        page_lines = pages_data[pnum-1]["lines"]
        for idx, (t, bbox) in enumerate(items, start=1):
            try:
                df = t.df
            except Exception:
                # Some versions: t.data is list of lists
                df = pd.DataFrame(getattr(t, "data", []))
            csv_path = os.path.join(outdir, f"page{pnum}_table{idx}.csv")
            df.to_csv(csv_path, index=False)
            # Camelot bbox may be None; approximate from table shape by scanning text lines
            if bbox is None and hasattr(t, "to_json"):
                try:
                    meta = json.loads(t.to_json())
                    bbox = tuple(meta["tables"][0]["bbox"])
                except Exception:
                    bbox = (0,0,0,0)
            caption = nearest_caption_for_bbox(page_lines, bbox, label="Table")
            tables.append({
                "page": pnum,
                "caption": caption,
                "bbox": bbox,
                "csv_path": csv_path,
                "source": "camelot"
            })

    # From pdfplumber (include any pages missing or as additional tables)
    for pnum, items in pl_tables.items():
        page_lines = pages_data[pnum-1]["lines"]
        for j, (df, bbox) in enumerate(items, start=1):
            csv_path = os.path.join(outdir, f"page{pnum}_pl_table{j}.csv")
            try:
                df.to_csv(csv_path, index=False, header=False)
            except Exception:
                # fallback if non-rectangular
                pd.DataFrame(df).to_csv(csv_path, index=False, header=False)
            caption = nearest_caption_for_bbox(page_lines, bbox, label="Table")
            tables.append({
                "page": pnum,
                "caption": caption,
                "bbox": bbox,
                "csv_path": csv_path,
                "source": "pdfplumber"
            })

    # Figures (images) + captions
    figures = []
    for p in pages_data:
        pnum = p["page"]
        page = doc[pnum-1]
        page_lines = p["lines"]
        for k, img in enumerate(p["images"], start=1):
            bbox = img["bbox"]
            img_path = os.path.join(outdir, f"page{pnum}_fig{k}.png")
            export_image_region(page, bbox, img_path, dpi=200)
            caption = find_caption_near_bbox(page_lines, bbox, direction="below") or \
                      find_caption_near_bbox(page_lines, bbox, direction="either")
            figures.append({
                "page": pnum,
                "caption": caption,
                "bbox": bbox,
                "image_path": img_path
            })

    # Build Markdown summary
    md_lines = []
    md_lines.append(f"# Summary for `{Path(pdf_path).name}`\n")
    md_lines.append("## Headings\n")
    if not headings:
        md_lines.append("_No headings detected by font-size heuristic._\n")
    else:
        for h in headings:
            indent = "  "*(h["level"]-1)
            md_lines.append(f"{indent}- **(p{h['page']})** {h['text']}  \n")
    md_lines.append("\n## Tables\n")
    if not tables:
        md_lines.append("_No tables detected._\n")
    else:
        for t in tables:
            cap = t["caption"] or "_(no caption found)_"
            md_lines.append(f"- (p{t['page']}) **{_norm(cap)}** → `{t['csv_path']}`\n")
    md_lines.append("\n## Figures\n")
    if not figures:
        md_lines.append("_No figures detected._\n")
    else:
        for f in figures:
            cap = f["caption"] or "_(no caption found)_"
            md_lines.append(f"- (p{f['page']}) **{_norm(cap)}** → `{f['image_path']}`\n")

    summary = {
        "pdf": str(pdf_path),
        "headings": headings,
        "tables": tables,
        "figures": figures,
    }

    json_path = os.path.join(outdir, "summary.json")
    md_path = os.path.join(outdir, "summary.md")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))

    return summary, json_path, md_path


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--pdf", default= "data/MC15 Deines Chiropractic.pdf", help="Path to PDF")
    ap.add_argument("-o", "--outdir", default="pdf_summary_out", help="Output directory")
    args = ap.parse_args()
    summary, jpath, mpath = summarize_pdf(args.pdf, args.outdir)
    print(f"Saved JSON → {jpath}")
    print(f"Saved Markdown → {mpath}")