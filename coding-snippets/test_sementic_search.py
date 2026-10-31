#!/usr/bin/env python3
import os
from dotenv import load_dotenv

from openai import OpenAI
import numpy

load_dotenv()

client = OpenAI(
    base_url=os.getenv("BASE_URL"),
    api_key=os.getenv("API_KEY")
)

def get_embedding(text):
    response = client.embeddings.create(
        model=os.getenv("EMBED_MODEL"),
        input=text
    )
    return response.data[0].embedding

def cosine_similarity(A,B):
    dot_prod = numpy.dot(A,B)
    norm_a = numpy.linalg.norm(A)
    norm_b = numpy.linalg.norm(B)
    return dot_prod / (norm_a * norm_b)

text1 = "Python is a programming language"
text2 = "Python is used to write AI related code snippets"
text3 = "I love driking tea every morning"

embedding1 = get_embedding(text1)
embedding2 = get_embedding(text2)
embedding3 = get_embedding(text3)

similary12 = cosine_similarity(embedding1,embedding2)
similary13 = cosine_similarity(embedding1,embedding3)


print("Similary b/w text1 and text2: ", f"{similary12:.2f}" )
print("Similary b/w text1 and text3: ", f"{similary13:.2f}" )