import os
from typing import List
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from dotenv import load_dotenv
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from difflib import SequenceMatcher
import traceback
from groq import Groq

# Load environment variables
load_dotenv()
API_KEYS = [
    os.getenv("GROQ_API_KEY_1"),
    os.getenv("GROQ_API_KEY_2"),
    os.getenv("GROQ_API_KEY_3"),
    os.getenv("GROQ_API_KEY_4"),
]
API_KEYS = [key for key in API_KEYS if key]  # Filter out None

CHROMA_DIR = "chroma_store"
EMBED_MODEL = "BAAI/bge-large-en-v1.5"
LLM_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"
MAX_TOKENS = 700

embedder = SentenceTransformer(EMBED_MODEL)
chroma_client = chromadb.PersistentClient(path=CHROMA_DIR, settings=Settings(anonymized_telemetry=False))

app = FastAPI(title="NBC RAG Assistant")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    collection_id: List[str]
    query: str
    top_k: int = 5

def call_groq_api(prompt: str):
    last_exception = None
    for api_key in API_KEYS:
        try:
            client = Groq(api_key=api_key)
            response = client.chat.completions.create(
                model=LLM_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_completion_tokens=MAX_TOKENS,
                top_p=1,
                stream=False
            )
            answer = response.choices[0].message.content.strip()
            if not answer:
                raise ValueError("Empty answer returned by LLM API.")
            return answer
        except Exception as e:
            print(f"Groq API key failed: {api_key[:8]}..., error: {e}")
            last_exception = e
    raise HTTPException(status_code=500, detail=f"All Groq API keys failed. Last error: {last_exception}")

@app.post("/chat")
async def chat_with_nbc(req: ChatRequest):
    collection_hits = {}
    for coll_id in req.collection_id:
        try:
            collection = chroma_client.get_collection(name=coll_id)
        except Exception:
            continue
        query_embedding = embedder.encode(req.query).tolist()
        try:
            results = collection.query(query_embeddings=[query_embedding], n_results=req.top_k)
        except Exception:
            continue
        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        def is_partial_match(query, text):
            return any(q in text.lower() for q in query.lower().split())
        def fuzzy_match(query, text, threshold=0.6):
            return SequenceMatcher(None, query.lower(), text.lower()).ratio() > threshold
        matched = []
        for doc, meta in zip(documents, metadatas):
            if is_partial_match(req.query, doc) or fuzzy_match(req.query, doc):
                matched.append((doc, meta))
        selected = matched if matched else list(zip(documents, metadatas))
        if selected:
            if coll_id not in collection_hits:
                collection_hits[coll_id] = []
            collection_hits[coll_id].extend(selected)
    if not collection_hits:
        return {
            "answer": "No relevant context found in any selected collections.\nCollection: Not applicable"
        }
    context_str = ""
    for coll, hits in collection_hits.items():
        context_str += f"Collection: {coll}\n"
        seen = set()
        index = 1
        for chunk, meta in hits:
            clause = meta.get("clause", "omit")
            page = meta.get("page", "omit")
            note = meta.get("note", None)
            key = (clause, page)
            if key in seen:
                continue
            seen.add(key)
            context_str += f"[{index}] Page {page} | Clause {clause}:\n{chunk.strip()}\n"
            if note:
                context_str += f"Note: {note}\n"
            index += 1
        context_str += "\n"
    collections_str = ", ".join(collection_hits.keys())
    prompt = f"""You are a senior building code consultant specializing in Indian and international building standards.
Your job is to answer user questions using only the provided context. You must ensure clarity, accuracy, and reference every answer to relevant clauses, pages, tables, and notes.
When the user input is a statement or unclear, use common sense to rephrase it into a proper, grammatically correct question.

{context_str}
Question: {req.query}
Collections: {collections_str}

#### Prompt Rules for Answering User Queries.####
### General Response Rules ###
1. If the user query is incomplete, fragmented, or written like a keyword phrase (e.g., “30 mtrs height mercantile building pressurization of staircase is required”), reframe it into a full, grammatically correct question before answering.
2. Always display the reframed question at the top under:
   Reframed Question: [Full question]
3. Use ONLY the provided context. Do not assume or fabricate details outside context.
4. If the context lacks a direct answer, but Table/Clause references can be inferred, guide the user to the correct Table/Clause explicitly.
5. Answer clearly and concisely in a professional tone. Use bullet points or steps if necessary.
6. Always include:
   Clause Number
   Page Number
7. If a figure or table is referenced, mention:
   Table/Figure number
   Its title or summary
8. If a Note is mentioned in the provided context or associated with a Table/Figure, include a clear explanation of the note under a section titled 'Note Explanation'. If there is no note, omit this section.
9. Do not include irrelevant details, unnecessary repetition, or friendly phrases. Keep answers factual and precise.
10. For partial keyword matches (e.g., “gym”, “hydrant”), expand it into the full matching entry (for example, “gym” → “Gym, stadium, play area” from Table 6-1).
11. If the answer is not available in the provided context, respond with:
    The provided context does not contain information relevant to this question.
### Special Rules for Tables and Clauses ###
12. If the query mentions phrases like “Table X”, “Clause Y”, “Size of Mains”, “Sprinkler Installation”, “Pressurization of Staircase”, or similar:
   Reframe the query to explicitly mention the corresponding Table/Clause.
   Search the context for references to that Table/Clause.
   If not found, state: “Table X / Clause Y is relevant to this query, but the provided context does not include its details.”
13. When the user query involves “Size of Mains” (directly or indirectly):
   Always assume Table 8 of NBC is relevant.
   Always reframe the question to include “automatic sprinkler installation”.
   Always include sizing details from Table 8, referring to building type and applicable heights.
   Reference Clause 5.1.1(a) and Page 312 in the answer.
   Do not skip answering if “automatic sprinkler installation” is not in the user’s query.
14. Query Handling:
    Use fuzzy + semantic match together (combine keyword + embedding similarity).
    Always prioritize exact clause/table/page matches before semantic or fuzzy matches. (Update)
    Synonym mapping: expand terms automatically (example: “gym” → “Gym, stadium, play area”).
    Maintain a glossary for standard-specific terms (hydrant = firefighting system, FAR = floor area ratio, etc.).
### Collections Policy ###
15. If the user selects one or more collections (e.g., "NBC_DATA"):
    Use only the selected collection for retrieval.
    Clearly mention the collection name in the response under the heading "Collection."
16. If the user does not select any collection:
    Search across all available collections.
    For each answer, explicitly mention the collection name it was retrieved from.
17. If no relevant content is found in any selected collection:
Respond with:
"Collection: Not applicable"
18. If partial or related references exist (e.g., term present but exact detail missing):
    Do NOT use the above fallback.
    Instead, guide the user by explicitly mentioning the closest matching Clause, Table, or Figure available and instruct them where to look for exact details.
### Multi-Hit and Formatting Policy###
19. If multiple relevant clauses or tables are found:
If they belong to the same clause, merge and summarize them into one answer.
If they belong to different clauses, list each one separately under clear sub-headings. (New)
20. When showing tables, always render them clearly with borders (using tabulate or structured formatting). (New)
21. Do not restate the reframed question in the Answer section. It should only appear once at the top. (New)
== Answer Formatting Must Always Follow This Structure ==
Clause: [Clause Number] omit if not applicable
Page: [Page Number] omit if not applicable
Answer:
[Clear and precise answer from context]
Note Explanation:
[Explain notes if any, omit if not applicable]
Reference:
Clause title – Page Number
Table/Figure – Title (if applicable) omit if not applicable
Collection:
[Collection name used, or omit if not applicable]
###Rules for Answering User Queries###
== Important Style Rules ==
Do not use parentheses, brackets, asterisks, dashes, or any markdown formatting
Use plain text only
omit if not applicable
Do not include greetings, small talk, or personal opinions
Be factual, precise, direct, and unambiguous.
Do not fabricate or assume details outside the provided context
Do not repeat or restate the user query in the answer section
Use professional and formal language suitable for technical consulting
### End of Rules ###
"""

    try:
        answer = call_groq_api(prompt)
        return {"answer": answer}
    except Exception as e:
        print("Exception in /chat:", e)
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"LLM request failed: {e}")


PDF_DIR = "Standards"

@app.get("/api/list-pdfs")
async def list_pdfs():
    try:
        pdfs = [f for f in os.listdir(PDF_DIR) if f.lower().endswith(".pdf")]
        return {"pdfs": pdfs}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list PDFs: {e}")

@app.get("/api/list-collections")
async def list_collections():
    try:
        collections = chroma_client.list_collections()
        collection_names = [coll.name for coll in collections]
        return {"collections": collection_names}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list collections: {e}")

@app.get("/pdfs/{filename}")
async def serve_pdf(filename: str):
    file_path = os.path.join(PDF_DIR, filename)
    if os.path.exists(file_path):
        return FileResponse(file_path, media_type="application/pdf")
    raise HTTPException(status_code=404, detail="File not found")
