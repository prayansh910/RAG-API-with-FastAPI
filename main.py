from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator
import ollama
import chromadb
from chromadb.utils.embedding_functions.ollama_embedding_function import (
    OllamaEmbeddingFunction,
)

app = FastAPI(title="Local RAG API", description="Ask questions over your own documents.", version="1.0.0")

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Connect to the same ChromaDB collection you built in Step 2
client = chromadb.PersistentClient(path=str(BASE_DIR / "chroma_db"))

ef = OllamaEmbeddingFunction(
    model_name="nomic-embed-text",
    url="http://localhost:11434",
)

collection = client.get_or_create_collection(
    name="personal_profile",
    embedding_function=ef,
)


#Define the expected shape of incoming data for the POST endpoint 
class DocumentSubmission(BaseModel):
    user_name: str = Field(min_length=1, max_length=100)
    content: str = Field(min_length=1, max_length=100_000)

    @field_validator("user_name", "content")
    @classmethod
    def strip_values(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("This field cannot be blank.")
        return value


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def home():
    return (STATIC_DIR / "index.html").read_text(encoding="utf-8")


@app.post("/documents")  # POST endpoint - accepts data in the request body
def add_document(submission: DocumentSubmission):
    chunks = [chunk.strip() for chunk in submission.content.split("\n\n") if chunk.strip()]
    if not chunks:
        raise HTTPException(status_code=400, detail="Document content must include at least one non-empty paragraph.")

    # Stable IDs plus cleanup make repeated submissions replace a user's profile.
    existing = collection.get(where={"user_name": submission.user_name}, include=[])
    ids = [f"{submission.user_name}-chunk{i}" for i in range(len(chunks))]
    stale_ids = [item_id for item_id in existing.get("ids", []) if item_id not in ids]
    try:
        collection.upsert(
            ids=ids,
            documents=chunks,
            metadatas=[
                {"source": "profile", "user_name": submission.user_name, "chunk_index": i}
                for i in range(len(chunks))
            ],
        )
    except Exception as exc:
        raise HTTPException(status_code=503, detail="The local embedding service is unavailable. Start Ollama and ensure nomic-embed-text is installed.") from exc
    if stale_ids:
        collection.delete(ids=stale_ids)

    return {
        "message": f"Added {len(chunks)} chunks for user '{submission.user_name}'.",
        "user_name": submission.user_name,
        "chunks_added": len(chunks),
    }


@app.get("/ask")
def ask(question: str = Query(min_length=1, max_length=2_000), user: str | None = Query(default=None, max_length=100)):
    question = question.strip()
    if not question:
        raise HTTPException(status_code=422, detail="Question cannot be blank.")
    if user is not None:
        user = user.strip()
        if not user:
            raise HTTPException(status_code=422, detail="User filter cannot be blank.")
    # Step 1: RETRIEVE - search ChromaDB for relevant chunks
    query_params = {
        "query_texts": [question],
        "n_results": 2,
    }

    # Restrict retrieval to a particular user when requested
    if user:
        query_params["where"] = {"user_name": user}

    try:
        results = collection.query(**query_params)
    except Exception as exc:
        raise HTTPException(status_code=503, detail="The local embedding service is unavailable. Start Ollama and ensure nomic-embed-text is installed.") from exc

    # Safely handle missing or empty retrieval results
    documents = results.get("documents") or []
    context_chunks = documents[0] if documents else []
    context = "\n\n".join(context_chunks)

    # Don't ask the LLM to answer when no context was retrieved
    if not context.strip():
        return {
            "question": question,
            "answer": "I couldn't find relevant information in the available documents.",
            "context_used": [],
            "filtered_by_user": user,
        }

    # Step 2: AUGMENT - explicitly restrict answers to supported facts
    augmented_prompt = f"""You are a question-answering assistant for a document-based RAG system.

Answer the question using ONLY the facts explicitly stated in the context below.

Rules:
1. Do not use outside knowledge or make assumptions.
2. Do not treat related information as proof of an unstated fact.
3. If the context does not explicitly answer the question, say:
   "The provided context does not contain that information."
4. Never invent personal preferences, opinions, experiences, or facts.
5. Keep the answer concise.

Context:
<context>
{context}
</context>

Question:
{question}

Answer:"""
    # Step 3: GENERATE - use the local language model
    try:
        response = ollama.chat(
            model="qwen2.5:0.5b",
            messages=[
                {"role": "system", "content": "Answer only with facts supported by the supplied context. If unsupported, say the context does not contain that information."},
                {"role": "user", "content": augmented_prompt},
            ],
            options={"temperature": 0},
        )
    except Exception as exc:
        raise HTTPException(status_code=503, detail="The local Ollama model is unavailable. Check that Ollama is running and the model is installed.") from exc

    # Step 4: RETURN - include the answer and retrieved evidence
    return {
        "question": question,
        "answer": response["message"]["content"],
        "context_used": context_chunks,
        "filtered_by_user": user,
    }
