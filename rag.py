#!/usr/bin/env python3
"""Chunk a local knowledge base, embed each chunk, and answer semantic queries."""

import json
import os

import numpy
from dotenv import load_dotenv
from openai import OpenAI

# Load local settings before creating the OpenAI client.
load_dotenv()

KB_FOLDER = os.getenv("KB_FOLDER")
EMBED_MODEL = os.getenv("EMBED_MODEL")
VECTOR_STORE_FILE = os.getenv("VECTOR_STORE_FILE")

# Fail fast if the required configuration is missing.
required_settings = {
    "KB_FOLDER": KB_FOLDER,
    "EMBED_MODEL": EMBED_MODEL,
    "VECTOR_STORE_FILE": VECTOR_STORE_FILE,
    "BASE_URL": os.getenv("BASE_URL"),
    "API_KEY": os.getenv("API_KEY"),
}
missing = [name for name, value in required_settings.items() if not value]
if missing:
    raise RuntimeError(f"Missing required environment variables: {', '.join(missing)}")

client = OpenAI(
    base_url=os.getenv("BASE_URL"),
    api_key=os.getenv("API_KEY"),
)


def get_embedding(text):
    """Create a vector embedding for a single text string."""
    response = client.embeddings.create(model=EMBED_MODEL, input=text)
    return response.data[0].embedding


def cosine_similarity(vector_a, vector_b):
    """Return the cosine similarity between two vectors."""
    dot_prod = numpy.dot(vector_a, vector_b)
    norm_a = numpy.linalg.norm(vector_a)
    norm_b = numpy.linalg.norm(vector_b)
    return dot_prod / (norm_a * norm_b)


def load_documents(folder=KB_FOLDER):
    """Read every .txt file from the configured knowledge-base directory."""
    documents = []

    for filename in sorted(os.listdir(folder)):
        if not filename.endswith(".txt"):
            continue

        path = os.path.join(folder, filename)
        with open(path, "r", encoding="utf-8") as file_handle:
            documents.append({"source": filename, "text": file_handle.read()})
    return documents


def add_embeddings(chunks):
    """Add an embedding vector to each chunk before searching."""
    for chunk in chunks:
        chunk["vector"] = get_embedding(chunk["text"])
    return chunks


def chunk_document(document):
    """Split a document into paragraph-sized chunks that can be embedded."""
    chunks = []
    chunk_id = 0
    for part in document["text"].split("\n\n"):
        part = part.strip()
        if part:
            chunks.append({
                "source": document["source"],
                "chunk_id": chunk_id,
                "text": part,
            })
            chunk_id += 1
    return chunks


def create_chunks():
    """Collect all local documents and split them into searchable chunks."""
    documents = load_documents()
    chunks = []
    for document in documents:
        chunks.extend(chunk_document(document))
    return chunks


def save_vector_store(chunks):
    """Persist the chunk metadata and embeddings as JSON for quick reuse."""
    with open(VECTOR_STORE_FILE, "w", encoding="utf-8") as file_handle:
        json.dump(chunks, file_handle, indent=4)


def load_vector_store():
    """Load previously persisted embeddings from disk."""
    with open(VECTOR_STORE_FILE, "r", encoding="utf-8") as file_handle:
        return json.load(file_handle)


def load_chunks():
    if os.path.exists(VECTOR_STORE_FILE):
        chunks = load_vector_store()
        print(f"Loaded {len(chunks)} chunks from the vector store file {VECTOR_STORE_FILE}")
    else:
        chunks = create_chunks()
        chunks = add_embeddings(chunks)
        save_vector_store(chunks)
        print(f"Created and embedded {len(chunks)} and saved to the vector store file {VECTOR_STORE_FILE}")
    return chunks

# Load the cached vectors once so each search only needs to embed its query.
chunks = load_chunks()


def search(query, top_k=3):
    """Find the top matches for a query by embedding similarity."""
    query_vector = get_embedding(query)
    results = []
    for chunk in chunks:
        score = cosine_similarity(query_vector, chunk["vector"])
        results.append(
            {
                "score": float(score),
                "source": chunk["source"],
                "chunk_id": chunk["chunk_id"],
                "text": chunk["text"],
                "size": len(chunk["text"]),
            }
        )
    results.sort(key=lambda record: record["score"], reverse=True)
    return results[:top_k]
