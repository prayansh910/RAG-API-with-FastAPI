from fastapi import FastAPI
import ollama
import chromadb
from chromadb.utils.embedding_functions.ollama_embedding_function import (OllamaEmbeddingFunction)

app = FastAPI() #Create the fastapi application

#connect with the same chromadb collection we built in build_knowledge_base.py
client = chromadb.PersistentClient(path="./chroma_db")

ef= OllamaEmbeddingFunction(
    model_name="nomic_embed_text",
    url="http://localhost:11434",
)

collection = client.get_or_create_collection(
    name="personal_profile",
    embedding_function=ef,
)



@app.get("/ask")  # This creates a GET endpoint at /ask
def ask(question: str):  # FastAPI automatically reads "question" from the URL query string
    # Step 1: RETRIEVE - search ChromaDB for the 2 most relevant chunks
    results = collection.query(
        query_texts=[question],  # ChromaDB converts this to a vector and finds similar chunks
        n_results=2,  # Return the top 2 matches
    )
    # Combine the matching chunks into a single string
    context = "\n\n".join(results["documents"][0])

    # Step 2: AUGMENT - build a prompt that includes the retrieved context
    augmented_prompt = f"""Use the following context to answer the question.
If the context doesn't contain relevant information, say so.

Context:
{context}

Question: {question}"""

    # Step 3: GENERATE - send the augmented prompt to the local LLM
    response = ollama.chat(
        model="qwen2.5:0.5b",
        messages=[{"role": "user", "content": augmented_prompt}],
    )

    # Return the answer along with the context so users can verify the source
    return {
        "question": question,
        "answer": response["message"]["content"],
        "context_used": results["documents"][0],
    }