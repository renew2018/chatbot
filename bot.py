import os
import sys
from dotenv import load_dotenv
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from openai import OpenAI

# === Load API Key ===
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# === Config ===
CHROMA_DIR = "chroma_store"
COLLECTION_NAME = "nbc_data"
EMBED_MODEL = "BAAI/bge-large-en-v1.5"
LLM_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"
MAX_TOKENS = 700

# === Load Clients ===
print(" Loading ChromaDB...")
chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
collection = chroma_client.get_collection(COLLECTION_NAME)

print(" Loading embedding model...")
embedder = SentenceTransformer(EMBED_MODEL)

llm_client = OpenAI(
    api_key=GROQ_API_KEY,
    base_url="https://api.groq.com/openai/v1"
)

# === Generate Answer ===
def generate_answer(query, top_k=20):
    query_embedding = embedder.encode(query).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )

    contexts = results["documents"][0]
    metadatas = results["metadatas"][0]

    context_str = ""
    for i, (chunk, meta) in enumerate(zip(contexts, metadatas)):
        clause = meta.get("clause", "N/A")
        page = meta.get("page", "Unknown")
        context_str += f"[{i+1}] Page {page} | Clause {clause}:\n{chunk.strip()}\n\n"

    prompt = f"""You are a senior building code consultant specializing in the National Building Code (NBC) of India 2016 Volume 1.

Your job is to answer user questions using only the provided NBC context. You must ensure clarity, accuracy, and reference every answer to relevant clauses and pages.

Follow these strict guidelines for every response:

1. ONLY use the context provided — do not guess, assume, or fabricate information.
2. Answer concisely and clearly, using bullet points or numbered steps if appropriate.
3. If applicable, include the **exact clause number** and page number where the answer is found.
4. If a figure or table is referenced, include:
   - Table/Figure number (e.g., "Table 4.3")
   - Its title or summary
5. If the context does **not** contain the answer, say:
   - *“The provided NBC context does not contain information relevant to this question.”*

🧱 Always format your answer in this structure:

---
Clause: [Clause number]

Page: [Page number]

Answer: 
[Clear, direct explanation using only context.]

Reference:  
- [Clause title]  
- [Table/Figure if applicable]  
---

 Tone: Professional, concise, fact-based — no opinions, filler, or friendly small talk.
Context:
{context_str}

Question: {query}
"""

    response = llm_client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=MAX_TOKENS
    )

    return response.choices[0].message.content.strip()

# === CLI ===
if __name__ == "__main__":
    print(" NBC RAG Assistant (Groq Claude-style, Ctrl+C to exit)")
    while True:
        try:
            query = input("You: ").strip()
            if not query:
                continue
            answer = generate_answer(query)
            print(f"\nAssistant: {answer}\n")
        except KeyboardInterrupt:
            print("\n Exiting. Goodbye!")
            sys.exit()
        except Exception as e:
            print(f" Error: {e}")
