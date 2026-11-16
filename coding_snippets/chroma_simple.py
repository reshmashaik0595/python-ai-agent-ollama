#!/usr/bin/env python3
import os
import chromadb
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    base_url=os.getenv("BASE_URL"),
    api_key=os.getenv("API_KEY")
)

EMDED_MODEL = os.getenv("EMBED_MODEL")

def get_embedding(text):
    resonse = client.embeddings.create(
        model=EMDED_MODEL,
        input=text
    )
    return resonse.data[0].embedding

chunks = {
     "c0" : "Leave Policy - Maternity Leave\nMaternity leave is 26 weeks of paid leave for the first two children",
     "c1" : "Medical Policy - Maternity Cover\nHospitalization for childbirth is covered up to 75,000 rupees.",
     "c2" : "Travel Expense - Reimbursement\nSubmit expense claims within 30 days of the trip. Approved claims are reimbursement",
     "c3" : "Tech Docs - Deployment Windows\nProduction deployment are allowed on Tuesdays and Thursdays between 11 AM and 4PM"
        
}

ids = list(chunks.keys())
documents = list(chunks.values())

embeddings = []
for s in documents:
    embedding = get_embedding(s)
    embeddings.append(embedding)

# Use a persistent local client so the collection survives process restarts.
chroma = chromadb.PersistentClient(path='./chromadbStorage')

zorvex_collection = chroma.get_or_create_collection(
    name='zorvex_tech',
    configuration={
        "hnsw":{ #Hierarchical Naviagtion Small World - Fast Search Algo [Graph Based Algo]
            "space":"cosine" # by-default l2
        }
    }
)

# Upsert keeps each stored passage associated with its stable ID and vector.
zorvex_collection.upsert(
    ids=ids,
    documents=documents,
    embeddings=embeddings
)

query = "When is the production deployment allowed ?"
query_embedding = get_embedding(query)

# Retrieve the single closest stored passage for the query vector.
results = zorvex_collection.query(
    query_embeddings=[query_embedding],
    n_results=1
)

print(results['documents'][0][0])