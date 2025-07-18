import uuid
import io
import os
import re
import json
import fitz
import pdfplumber
import pytesseract
from PIL import Image
from pathlib import Path
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from chromadb import PersistentClient
from sentence_transformers import SentenceTransformer

# === Constants ===
EMBED_MODEL = "BAAI/bge-large-en-v1.5"
CHROMA_DIR = "chroma_store"
OUTPUT_DIR = "output"
COLLECTION_NAME = "nbc_data"

# === App & Initialization ===
app = FastAPI()
os.makedirs(OUTPUT_DIR, exist_ok=True)

model = SentenceTransformer(EMBED_MODEL)
client = PersistentClient(path=CHROMA_DIR)


# === Utilities ===
def clean_paragraphs(text):
    lines = text.split("\n")
    return [
        line.strip() for line in lines
        if line.strip() and not re.search(r"Supply Bureau.*valid upto", line)
    ]


def extract_clause_blocks(text):
    clause_pattern = re.compile(r'(?<!\d)(\d{1,2}(?:\.\d+)+)\s+([A-Z][^\n]{5,})')
    matches = list(clause_pattern.finditer(text))
    blocks = []
    for i, match in enumerate(matches):
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        clause_number = match.group(1)
        clause_title = match.group(2).strip()
        paragraphs = clean_paragraphs(text[start:end])
        blocks.append({
            "clause_number": clause_number,
            "clause_title": clause_title,
            "paragraphs": paragraphs
        })
    return blocks


def find_table_title(lines, index):
    for i in range(index - 1, max(index - 5, -1), -1):
        line = lines[i].strip()
        if re.match(r'^Table\s*\d+', line, re.IGNORECASE):
            return line
    return "Auto-detected Table"


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


def extract_figures(text):
    figures = re.findall(r'(Fig(?:ure)?\.?\s*\d+[^:\n]*)', text, re.IGNORECASE)
    result = []
    for fig in figures:
        parts = fig.split(None, 2)
        if len(parts) >= 3:
            result.append({"figure_number": parts[1].strip("."), "title": parts[2].strip()})
        elif len(parts) == 2:
            result.append({"figure_number": parts[1].strip("."), "title": ""})
    return result


def ocr_text_from_fitz_page(page):
    pix = page.get_pixmap(dpi=300)
    img = Image.open(io.BytesIO(pix.tobytes("png")))
    return pytesseract.image_to_string(img)


def process_pdf(pdf_path: str):
    doc = fitz.open(pdf_path)
    structured_data = []
    for page_num, page in enumerate(doc):
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


def embed_and_store(data, collection_name: str):
    collection = client.get_or_create_collection(name=collection_name)

    for doc in data:
        text_parts = []
        if doc.get("clause_number"):
            text_parts.append(f"Clause {doc['clause_number']}: {doc.get('clause_title', '')}")
        text_parts += [p.strip() for p in doc.get("paragraphs", []) if p.strip()]

        for table in doc.get("tables", []):
            if table.get("title"):
                text_parts.append(f"Table: {table['title']}")
            if "columns" in table:
                text_parts.append(" | ".join(table["columns"]))
            text_parts += [" | ".join(row) for row in table.get("rows", [])]
            text_parts += [f"Note: {note}" for note in table.get("notes", [])]

        for fig in doc.get("figures", []):
            text_parts.append(f"Figure {fig.get('figure_number')}: {fig.get('title', '')}")

        full_text = " ".join(text_parts).strip()
        if not full_text:
            continue

        embedding = model.encode(full_text).tolist()
        metadata = {
            "page": doc.get("page", 0),
            "clause": doc.get("clause_number", ""),
            "title": doc.get("clause_title", "")
        }

        collection.add(
            documents=[full_text],
            metadatas=[metadata],
            embeddings=[embedding],
            ids=[str(uuid.uuid4())]
        )
    return collection.count()


# === API Routes ===

@app.post("/upload_pdf")
async def upload_pdf(file: UploadFile = File(...)):
    filename = file.filename
    temp_pdf_path = f"temp_{uuid.uuid4()}.pdf"

    with open(temp_pdf_path, "wb") as f:
        f.write(await file.read())

    try:
        # Process & store
        structured_data = process_pdf(temp_pdf_path)
        json_filename = f"{Path(filename).stem}.json"
        json_path = os.path.join(OUTPUT_DIR, json_filename)

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(structured_data, f, indent=2, ensure_ascii=False)

        count = embed_and_store(structured_data, collection_name=COLLECTION_NAME)

        return JSONResponse({
            "message": f"✅ Processed '{filename}' and added {count} entries to ChromaDB collection '{COLLECTION_NAME}'",
            "collection": COLLECTION_NAME,
            "count": count,
            "output_json": json_filename
        })
    finally:
        os.remove(temp_pdf_path)


@app.get("/pdf_data/{file_id}")
def get_pdf_data(file_id: str):
    json_path = os.path.join(OUTPUT_DIR, f"{file_id}.json")
    if not os.path.exists(json_path):
        return JSONResponse({"error": "File not found"}, status_code=404)
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


@app.put("/update_pdf/{file_id}")
def update_pdf(file_id: str):
    json_path = os.path.join(OUTPUT_DIR, f"{file_id}.json")
    if not os.path.exists(json_path):
        return JSONResponse({"error": "File not found"}, status_code=404)
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    count = embed_and_store(data, COLLECTION_NAME)
    return {"message": f"✅ Updated embeddings in '{COLLECTION_NAME}' from '{file_id}.json'", "updated_count": count}


@app.delete("/delete_pdf/{file_id}")
def delete_pdf(file_id: str):
    json_path = os.path.join(OUTPUT_DIR, f"{file_id}.json")
    if os.path.exists(json_path):
        os.remove(json_path)
    return {
        "message": f"Deleted JSON '{file_id}.json'. Note: Embeddings are still in ChromaDB unless deleted separately."}
