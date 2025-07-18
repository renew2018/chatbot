"""import re
import fitz  # PyMuPDF
import pdfplumber
import pytesseract
from PIL import Image
import io
import json
from pathlib import Path
from tqdm import tqdm

PDF_PATH = "data/NBC 2016 Vol 1.PDF"
OUTPUT_FILE = "output/nbc_full_data.json"

def clean_paragraphs(text):
    lines = text.split("\n")
    cleaned = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        # Remove watermarks/footers
        if " Supply Bureau Under the License from BIS for LARSEN AND TOUBRO CONSTRUCTION - MANAPAKKAM, CHENNAI ON 17-03-2017 08:57:36 (123.63.24.35) valid upto31-12-2016" in line or "valid upto" in line:
            continue
        cleaned.append(line)
    return cleaned

def extract_clause_blocks(text):
    clause_pattern = re.compile(r'(?<!\d)(\d{1,2}(?:\.\d+)+)\s+([A-Z][^\n]{5,})')
    matches = list(clause_pattern.finditer(text))
    blocks = []
    for i, match in enumerate(matches):
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        clause_number = match.group(1)
        clause_title = match.group(2).strip()
        paragraph_text = text[start:end]
        paragraphs = clean_paragraphs(paragraph_text)
        blocks.append({
            "clause_number": clause_number,
            "clause_title": clause_title,
            "paragraphs": paragraphs
        })
    return blocks

def extract_tables_from_page(pdf_path, page_number):
    try:
        with pdfplumber.open(pdf_path) as pdf:
            if page_number < len(pdf.pages):
                page = pdf.pages[page_number]
                raw_tables = page.extract_tables()
                formatted = []
                for tbl in raw_tables:
                    if not tbl or len(tbl) < 2:
                        continue
                    header, *rows = tbl
                    formatted.append({
                        "title": f"Table on Page {page_number + 1}",
                        "columns": header,
                        "rows": rows,
                        "notes": []
                    })
                return formatted
    except Exception:
        return []
    return []

def extract_figures(text):
    figures = re.findall(r'(Fig(?:ure)?\.?\s*\d+[^:\n]*)', text, re.IGNORECASE)
    result = []
    for fig in figures:
        parts = fig.split(None, 2)
        if len(parts) >= 3:
            result.append({
                "figure_number": parts[1].strip("."), "title": parts[2].strip()
            })
        elif len(parts) == 2:
            result.append({
                "figure_number": parts[1].strip("."), "title": ""
            })
    return result

def ocr_text_from_fitz_page(page):
    pix = page.get_pixmap(dpi=300)
    img = Image.open(io.BytesIO(pix.tobytes("png")))
    text = pytesseract.image_to_string(img)
    return text

def process_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    structured_data = []

    for page_num in tqdm(range(len(doc)), desc="📄 Parsing PDF"):
        page = doc[page_num]
        try:
            text = page.get_text("text")
        except Exception:
            text = ""

        if not text or len(text.strip()) < 20:
            text = ocr_text_from_fitz_page(page)

        text_cleaned = "\n".join(clean_paragraphs(text))
        clause_blocks = extract_clause_blocks(text_cleaned)
        tables = extract_tables_from_page(pdf_path, page_num)
        figures = extract_figures(text_cleaned)

        if clause_blocks:
            for block in clause_blocks:
                block.update({
                    "page": page_num + 1,
                    "tables": tables,
                    "figures": figures
                })
                structured_data.append(block)
        else:
            structured_data.append({
                "clause_number": "",
                "clause_title": "",
                "page": page_num + 1,
                "paragraphs": clean_paragraphs(text_cleaned),
                "tables": tables,
                "figures": figures
            })

    return structured_data

if __name__ == "__main__":
    Path("output").mkdir(exist_ok=True)
    structured = process_pdf(PDF_PATH)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(structured, f, indent=2, ensure_ascii=False)
    print(f"✅ Done. Output saved to {OUTPUT_FILE}")
"""
import re
import fitz  # PyMuPDF
import pdfplumber
import pytesseract
from PIL import Image
import io
import json
from pathlib import Path
from tqdm import tqdm

PDF_PATH = "data/NBC 2016 Vol 1.PDF"
OUTPUT_FILE = "output/nbc_full_data.json"

# === Clean Paragraphs and Remove Footer ===
def clean_paragraphs(text):
    lines = text.split("\n")
    cleaned = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        # Remove known footer pattern
        if re.search(r"Supply Bureau.*valid upto", line):
            continue
        cleaned.append(line)
    return cleaned

# === Clause Block Extraction (Hierarchical Style) ===
def extract_clause_blocks(text):
    clause_pattern = re.compile(r'(?<!\d)(\d{1,2}(?:\.\d+)+)\s+([A-Z][^\n]{5,})')
    matches = list(clause_pattern.finditer(text))
    blocks = []
    for i, match in enumerate(matches):
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        clause_number = match.group(1)
        clause_title = match.group(2).strip()
        paragraph_text = text[start:end]
        paragraphs = clean_paragraphs(paragraph_text)
        blocks.append({
            "clause_number": clause_number,
            "clause_title": clause_title,
            "paragraphs": paragraphs
        })
    return blocks

# === Table Title Detection (lines above table) ===
def find_table_title(lines, index):
    for i in range(index - 1, max(index - 5, -1), -1):
        line = lines[i].strip()
        if re.match(r'^Table\s*\d+', line, re.IGNORECASE):
            return line
    return "Auto-detected Table"

# === Extract Tables with Titles ===
def extract_tables_from_page(pdf_path, page_number):
    tables = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            if page_number < len(pdf.pages):
                page = pdf.pages[page_number]
                lines = page.extract_text().split("\n") if page.extract_text() else []

                raw_tables = page.extract_tables()
                for idx, tbl in enumerate(raw_tables):
                    if not tbl or len(tbl) < 2:
                        continue
                    title = find_table_title(lines, idx)
                    header, *rows = tbl
                    tables.append({
                        "title": title,
                        "columns": [c.strip() if c else "" for c in header],
                        "rows": [[c.strip() if c else "" for c in row] for row in rows],
                        "notes": []
                    })
    except Exception:
        pass
    return tables

# === Extract Figure Captions (e.g., Fig. 5 Title) ===
def extract_figures(text):
    figures = re.findall(r'(Fig(?:ure)?\.?\s*\d+[^:\n]*)', text, re.IGNORECASE)
    result = []
    for fig in figures:
        parts = fig.split(None, 2)
        if len(parts) >= 3:
            result.append({
                "figure_number": parts[1].strip("."), "title": parts[2].strip()
            })
        elif len(parts) == 2:
            result.append({
                "figure_number": parts[1].strip("."), "title": ""
            })
    return result

# === Fallback OCR using PyTesseract ===
def ocr_text_from_fitz_page(page):
    pix = page.get_pixmap(dpi=300)
    img = Image.open(io.BytesIO(pix.tobytes("png")))
    text = pytesseract.image_to_string(img)
    return text

# === Process Full PDF ===
def process_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    structured_data = []

    for page_num in tqdm(range(len(doc)), desc="📄 Parsing PDF"):
        page = doc[page_num]
        try:
            text = page.get_text("text")
        except Exception:
            text = ""

        # OCR fallback
        if not text or len(text.strip()) < 20:
            text = ocr_text_from_fitz_page(page)

        text_cleaned = "\n".join(clean_paragraphs(text))
        clause_blocks = extract_clause_blocks(text_cleaned)
        tables = extract_tables_from_page(pdf_path, page_num)
        figures = extract_figures(text_cleaned)

        if clause_blocks:
            for block in clause_blocks:
                block.update({
                    "page": page_num + 1,
                    "tables": tables,
                    "figures": figures
                })
                structured_data.append(block)
        else:
            structured_data.append({
                "clause_number": "",
                "clause_title": "",
                "page": page_num + 1,
                "paragraphs": clean_paragraphs(text_cleaned),
                "tables": tables,
                "figures": figures
            })

    return structured_data

# === Save to JSON ===
if __name__ == "__main__":
    Path("output").mkdir(exist_ok=True)
    structured = process_pdf(PDF_PATH)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(structured, f, indent=2, ensure_ascii=False)
    print(f" Done. Output saved to {OUTPUT_FILE}")
