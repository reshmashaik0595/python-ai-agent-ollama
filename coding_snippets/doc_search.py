#!/usr/bin/env python3
"""Simple document-search example that embeds text files and finds relevant matches."""

import os

import numpy
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables before contacting the embedding provider.
load_dotenv()

KB_FOLDER = os.getenv("KB_FOLDER")
EMBED_MODEL = os.getenv("EMBED_MODEL")

# Fail fast if required configuration is missing.
required_settings = {
    "KB_FOLDER": KB_FOLDER,
    "EMBED_MODEL": EMBED_MODEL,
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
    """Create an embedding vector for a string using the configured model."""
    response = client.embeddings.create(model=EMBED_MODEL, input=text)
    return response.data[0].embedding


def cosine_similarity(vector_a, vector_b):
    """Calculate cosine similarity between two vectors."""
    dot_prod = numpy.dot(vector_a, vector_b)
    norm_a = numpy.linalg.norm(vector_a)
    norm_b = numpy.linalg.norm(vector_b)
    return dot_prod / (norm_a * norm_b)


def load_documents(folder=KB_FOLDER):
    """Read all .txt documents from the knowledge-base directory."""
    documents = []

    for filename in sorted(os.listdir(folder)):
        if not filename.endswith(".txt"):
            continue

        path = os.path.join(folder, filename)
        with open(path, "r", encoding="utf-8") as file_handle:
            documents.append({"source": filename, "text": file_handle.read()})
    return documents


def add_embeddings(documents):
    """Attach an embedding vector to each document for similarity search."""
    for document in documents:
        document["vector"] = get_embedding(document["text"])
    return documents


def search(query, documents, top_k=3):
    """Return the most relevant documents for a search query."""
    query_vector = get_embedding(query)
    results = []
    for document in documents:
        score = cosine_similarity(query_vector, document["vector"])
        results.append(
            {
                "score": float(score),
                "source": document["source"],
                "text": document["text"],
                "size": len(document["text"]),
            }
        )
    results.sort(key=lambda record: record["score"], reverse=True)
    return results[:top_k]


def test(user_query, documents, top_k=3):
    """Print the highest-scoring matches for a sample search."""
    print(f"\nUser Query: {user_query}\n")
    results = search(user_query, documents, top_k)

    for result in results:
        print(f"{result['score']:.2f} - {result['source']} - {result['size']} characters")


# Embed every document once, then search the vectorized knowledge base.
documents = load_documents()
documents = add_embeddings(documents)
print(f"We have embedded {len(documents)} documents")

user_query = "How many days of maternity leave are allowed?"
test(user_query, documents, 3)