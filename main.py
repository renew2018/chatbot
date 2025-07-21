import os
import io
import re
import uuid
import json
import fitz
import pdfplumber
import pytesseract
import nltk
from PIL import Image
from pathlib import Path
from typing import List

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Depends
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from dotenv import load_dotenv
import secrets

from sentence_transformers import SentenceTransformer
from chromadb import PersistentClient

nltk.download("punkt")

# === Load environment ===
load_dotenv()
security = HTTPBasic()

# === Config ===
CHROMA_DIR = "chroma_store"
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# === App ===
app = FastAPI(
    title="Extractor and Embed",
    description="Upload PDFs and extract structured content with embeddings (Auth required)",
    version="2.0"
)

# === Embedding Model & ChromaDB ===
model = SentenceTransformer("BAAI/bge-large-en-v1.5")
client = PersistentClient(path=CHROMA_DIR)

# === Authentication ===
def verify_user(credentials: HTTPBasicCredentials = Depends(security)):
    username = credentials.username
    password = credentials.password

    env_user = os.getenv("USERNAME")
    env_pass = os.getenv("PASSWORD")

    if env_user and env_pass:
        valid = secrets.compare_digest(username, env_user) and secrets.compare_digest(password, env_pass)
    else:
        valid = False

    if not valid:
        raise HTTPException(
            status_code=401,
            detail="Invalid credentials",
            headers={"WWW-Authenticate": "Basic"},
        )
    return username

# === Utility Functions ===

def clean_paragraphs(text):
    lines = text.split("\n")
    return [line.strip() for line in lines if line.strip() and not re.search(r"Supply Bureau.*valid upto", line)]

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

def process_pdf(pdf_path: str) -> List[dict]:
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

def embed_and_store(data: List[dict], collection_name: str) -> int:
    collection = client.get_or_create_collection(name=collection_name)
    count = 0

    for doc in data:
        parts = []

        if doc.get("clause_number"):
            parts.append(f"Clause {doc['clause_number']}: {doc.get('clause_title', '')}")
        parts += doc.get("paragraphs", [])

        for table in doc.get("tables", []):
            parts.append(f"Table: {table['title']}")
            if "columns" in table:
                parts.append(" | ".join(table["columns"]))
            parts += [" | ".join(row) for row in table.get("rows", [])]
            parts += [f"Note: {n}" for n in table.get("notes", [])]

        for fig in doc.get("figures", []):
            parts.append(f"Figure {fig.get('figure_number')}: {fig.get('title', '')}")

        full_text = " ".join(parts).strip()
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
        count += 1

    return count

# === Endpoints ===

@app.get("/collections", summary="List ChromaDB collections")
def list_collections(user: str = Depends(verify_user)):
    return [c.name for c in client.list_collections()]

@app.post("/upload_pdf", summary="Upload PDF and embed to ChromaDB")
async def upload_pdf(
    file: UploadFile = File(...),
    collection_name: str = Form(...),
    user: str = Depends(verify_user)
):
    filename = file.filename
    temp_path = f"temp_{uuid.uuid4()}.pdf"
    with open(temp_path, "wb") as f:
        f.write(await file.read())

    try:
        structured_data = process_pdf(temp_path)
        json_filename = f"{Path(filename).stem}.json"
        json_path = os.path.join(OUTPUT_DIR, json_filename)

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(structured_data, f, indent=2, ensure_ascii=False)

        count = embed_and_store(structured_data, collection_name)

        return JSONResponse({
            "message": f"Processed '{filename}' and embedded {count} entries to '{collection_name}'",
            "collection": collection_name,
            "count": count,
            "output_json": json_filename
        })
    finally:
        os.remove(temp_path)

@app.delete("/delete_collection/{collection_name}", summary=" Delete ChromaDB collection")
def delete_chroma_collection(
    collection_name: str,
    user: str = Depends(verify_user)
):
    try:
        client.delete_collection(name=collection_name)
        return {"message": f"Deleted collection '{collection_name}'"}
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Failed to delete collection: {str(e)}")
