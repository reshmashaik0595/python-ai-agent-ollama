#!/usr/bin/env python3
import os
from openai import OpenAI
from dotenv import load_dotenv
import numpy
load_dotenv()
import json

KB_FOLDER = os.getenv('KB_FOLDER')
EMBED_MODEL = os.getenv("EMBED_MODEL")
VECTOR_STORE_FILE = os.getenv("VECTOR_STORE_FILE")

client = OpenAI(
    base_url=os.getenv("BASE_URL"),
    api_key=os.getenv("API_KEY")
)

def get_embedding(text):
    response = client.embeddings.create(
        model=EMBED_MODEL,
        input=text
    )
    return response.data[0].embedding

def cosine_similarity(A,B):
    dot_prod = numpy.dot(A,B)
    norm_a = numpy.linalg.norm(A)
    norm_b = numpy.linalg.norm(B)
    return dot_prod / (norm_a * norm_b)

#Load the documents
def load_documents(folder=KB_FOLDER):
    documents = []

    for filename in sorted(os.listdir(folder)):
        if not filename.endswith(".txt"):
            continue

        path = os.path.join(folder, filename)
        with open(path,"r",encoding="utf-8") as f:
            documents.append(
                {
                    "source":filename,
                    "text":f.read()
                }
            )
    return documents

def add_embeddings(chunks):
    for chunk in chunks:
        vector = get_embedding(chunk["text"])
        chunk["vector"] = vector
    return chunks

def search(query, chunks, top_k=3):
    query_vector = get_embedding(query)
    results = []
    for chunk in chunks:
        score = cosine_similarity(query_vector, chunk["vector"])
        results.append(
            {
                "score":float(score),
                "source":chunk["source"],
                "chunk_id":chunk["chunk_id"],
                "text":chunk["text"],
                "size":len(chunk["text"])
            }
        )
    results.sort(key=lambda rec:rec["score"],reverse=True)
    return results[:top_k]

def chunk_document(document):
    chunks = []
    chunk_id = 0
    for part in document["text"].split("\n\n"):
        part = part.strip()
        if part:
            chunks.append(
                {
                    "source":document["source"],
                    "chunk_id":chunk_id,
                    "text":part
                }
            )
            chunk_id = chunk_id + 1
    return chunks   

def create_chunks():
    documents = load_documents()
    chunks = []
    for document in documents:
        chunks.extend(chunk_document(document))
    return chunks

def test(user_query, chunks, top_k=3):
    print(f"\nUser Query: {user_query}\n")
    results = search(user_query,chunks,top_k )

    for r in results:
        print(f"{r["score"]:.2f} - {r["source"]} - {r["size"]} characters")

def save_vector_store(chunks):
    with open(VECTOR_STORE_FILE,"w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=4)

def load_vector_store():
    with open(VECTOR_STORE_FILE,"r",encoding="utf-8") as f:
        return json.load(f)

chunks = []
if os.path.exists(VECTOR_STORE_FILE):
    chunks = load_vector_store()
    print(f"Loaded {len(chunks)} chunks from the vector store file {VECTOR_STORE_FILE}")
else:
    chunks = create_chunks()
    chunks = add_embeddings(chunks)
    save_vector_store(chunks)
    print(f"Created and embedded {len(chunks)} and saved to the vector store file {VECTOR_STORE_FILE}")

chunks = create_chunks()
chunks = add_embeddings(chunks)
print(f"We have embedded {len(chunks)} chunks from our knowledge base.")

user_query = "What are the benifits of maternity leave"
test(user_query, chunks, 3)