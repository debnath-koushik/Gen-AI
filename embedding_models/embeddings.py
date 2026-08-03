from langchain_huggingface import HuggingFaceEmbeddings
from langchain_mistralai import MistralAIEmbeddings

import os
from dotenv import load_dotenv

# used this code to change default download location for store hugging face model data
os.environ['HF_HOME'] = 'D:/Koushik/huggingface_cache'

load_dotenv()

# HuggingFace enbedding code
# embedding = HuggingFaceEmbeddings(
#     model_name="sentence-transformers/all-MiniLM-L6-v2"
# )

# Mistral AI embedding code
embedding = MistralAIEmbeddings(
    model="mistral-embed"
)

texts = [
    "Hello this is my Machine",
    "I am using vscode for coding"
]

vector = embedding.embed_documents(texts)

print(vector)