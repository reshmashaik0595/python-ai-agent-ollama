#!/usr/bin/env python3
from openai import OpenAI
from dotenv import load_dotenv

import os
load_dotenv() #By default looks for .env file

client = OpenAI(
    base_url=os.getenv("BASE_URL"),
    api_key=os.getenv("API_KEY")
)

# A single request embeds both input strings and returns a vector for each one.
response = client.embeddings.create(
    model=os.getenv("EMBED_MODEL"),
    input=["I am good girl","I am a good coder"]
)

# Print each vector's size and a short preview rather than the full vector.
for i, emb in enumerate(response.data):
    print(f"Embedding {i} length: {len(emb.embedding)}")
    print(emb.embedding[:5])