import os
from typing import List
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from dotenv import load_dotenv
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from openai import OpenAI

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Constants
CHROMA_DIR = "chroma_store"
EMBED_MODEL = "BAAI/bge-large-en-v1.5"
LLM_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"
MAX_TOKENS = 700

# Load model and DB
print("Loading embedding model...")
embedder = SentenceTransformer(EMBED_MODEL)

print("Loading Groq LLM client...")
llm_client = OpenAI(
    api_key=GROQ_API_KEY,
    base_url="https://api.groq.com/openai/v1"
)

print("Loading ChromaDB...")
chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)

# Initialize FastAPI
app = FastAPI(title="NBC RAG Assistant")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change to specific domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def serve_index():
    return FileResponse("static/index.html")

# Request schema
class ChatRequest(BaseModel):
    collection_id: str
    query: str
    top_k: int = 20


# Chat endpoint
@app.post("/chat")
def chat_with_nbc(req: ChatRequest):
    try:
        collection = chroma_client.get_collection(name=req.collection_id)
    except Exception:
        raise HTTPException(status_code=404, detail=f"Collection '{req.collection_id}' not found.")

    query_embedding = embedder.encode(req.query).tolist()

    try:
        results = collection.query(query_embeddings=[query_embedding], n_results=req.top_k)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"ChromaDB query failed: {e}")

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]

    if not documents:
        return {"answer": "No relevant context found for your query."}

    context_str = ""
    for i, (chunk, meta) in enumerate(zip(documents, metadatas)):
        clause = meta.get("clause", "N/A")
        page = meta.get("page", "Unknown")
        context_str += f"[{i+1}] Page {page} | Clause {clause}:\n{chunk.strip()}\n\n"

    prompt = f"""You are a senior building code consultant specializing in the National Building Code (NBC) of India 2016 Volume's.

Your job is to answer user questions using only the provided NBC context. You must ensure clarity, accuracy, and reference every answer to relevant clauses and pages.

Follow these strict guidelines for every response:

1. ONLY use the context provided — do not guess, assume, or fabricate information.
2. Answer concisely and clearly, using bullet points or numbered steps if appropriate.
3. If applicable, include the **exact clause number** and page number where the answer is found.
4. If a figure or table is referenced, include:
   - Table/Figure number (e.g., "Table 4.3")
   - Its title or summary
5. If the context does **not** contain the answer, say:
   - “The provided NBC context does not contain information relevant to this question.”

🧱 Always format your answer in this structure:

---
Clause: [Clause number]

Page: [Page number]

Answer: 
[Clear, direct explanation using only context.]

Reference:  
- [Clause title] | [Page number]
- [Table/Figure if applicable]  
---

Tone: Professional, concise, fact-based — no opinions, filler, or friendly small talk.

Context:
{context_str}

Question: {req.query}
"""

    try:
        response = llm_client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=MAX_TOKENS
        )
        return {"answer": response.choices[0].message.content.strip()}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM failed: {e}")
