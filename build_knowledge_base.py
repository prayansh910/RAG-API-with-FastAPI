from pathlib import Path

import chromadb
from chromadb.utils.embedding_functions.ollama_embedding_function import OllamaEmbeddingFunction

BASE_DIR = Path(__file__).resolve().parent
profile_path = BASE_DIR / "profile.txt"
if not profile_path.exists():
    raise SystemExit("profile.txt not found. Create it with your own text, or submit text through the web app.")

text = profile_path.read_text(encoding="utf-8")
chunks = [chunk.strip() for chunk in text.split("\n\n") if chunk.strip()]
if not chunks:
    raise SystemExit("profile.txt is empty. Add at least one paragraph first.")

print(f"Loaded {len(chunks)} chunks from profile.txt")


# Initialize ChromaDB - PersistentClient saves data to disk so it survives restarts
client = chromadb.PersistentClient(path=str(BASE_DIR / "chroma_db"))

# Connect to Ollama's embedding model to convert text into vectors
ef = OllamaEmbeddingFunction(
    model_name="nomic-embed-text",
    url="http://localhost:11434",  # Ollama's default local address
)

# Create (or reuse) a collection - like a table in a database
collection = client.get_or_create_collection(
    name="personal_profile",
    embedding_function=ef,  # Tells ChromaDB how to convert text to vectors
)


# Add chunks to the collection - ChromaDB automatically generates embeddings
ids = [f"local-profile-chunk{i}" for i in range(len(chunks))]
existing = collection.get(where={"user_name": "local-profile"}, include=[])
stale_ids = [item_id for item_id in existing.get("ids", []) if item_id not in ids]
collection.upsert(
    ids=ids,
    documents=chunks,  # The actual text content
    metadatas=[{"source": "profile", "user_name": "local-profile", "chunk_index": i} for i in range(len(chunks))],
)
if stale_ids:
    collection.delete(ids=stale_ids)

print(f"Added {len(chunks)} chunks to the 'personal_profile' collection.")
print("Knowledge base built successfully!")

