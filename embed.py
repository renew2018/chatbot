import json
import uuid
from tqdm import tqdm
from chromadb import PersistentClient
from sentence_transformers import SentenceTransformer

# === Config ===
JSON_PATH = "output/nbc_full_data.json"
CHROMA_DIR = "chroma_store"
COLLECTION_NAME = "nbc_data"
EMBED_MODEL = "BAAI/bge-large-en-v1.5"

# === Load Embedding Model ===
print(" Loading embedding model...")
model = SentenceTransformer(EMBED_MODEL)

# === Setup ChromaDB ===
print(" Connecting to ChromaDB...")
client = PersistentClient(path=CHROMA_DIR)

# Delete and recreate collection (to match new vector size)
if COLLECTION_NAME in [c.name for c in client.list_collections()]:
    client.delete_collection(name=COLLECTION_NAME)

collection = client.get_or_create_collection(name=COLLECTION_NAME)

# === Load Extracted JSON Data ===
print(f" Loading data from {JSON_PATH}...")
with open(JSON_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

# === Embed and Store ===
print(" Embedding and storing...")
for doc in tqdm(data, desc=" Embedding"):
    text_parts = []

    if doc.get("clause_number"):
        text_parts.append(f"Clause {doc['clause_number']}: {doc.get('clause_title', '')}")

    for para in doc.get("paragraphs", []):
        if para.strip():
            text_parts.append(para.strip())

    for table in doc.get("tables", []):
        if table.get("title"):
            text_parts.append(f"Table: {table['title']}")
        if "columns" in table:
            text_parts.append(" | ".join(table["columns"]))
        for row in table.get("rows", []):
            text_parts.append(" | ".join(row))
        for note in table.get("notes", []):
            text_parts.append(f"Note: {note}")

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

print(f" Done! Stored {collection.count()} embedded documents.")
