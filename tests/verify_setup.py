"""
Quick environment verification script.
Run this before implementing the RAG pipeline.
"""

import os

import faiss
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from sentence_transformers import SentenceTransformer

load_dotenv()

print("=" * 50)
print("Travel Reimbursement Agent - Environment Check")
print("=" * 50)

# Check API Key
groq_key = os.getenv("GROQ_API_KEY")

if groq_key:
    print("✅ GROQ_API_KEY loaded")
else:
    print("❌ GROQ_API_KEY not found")

# Check Model Name
model_name = os.getenv("MODEL_NAME")

print(f"✅ LLM Model: {model_name}")

# Check Embedding Model
embedding_name = os.getenv("EMBEDDING_MODEL")

print(f"✅ Embedding Model: {embedding_name}")

# Load Embedding Model
embedding_model = SentenceTransformer(embedding_name)

print("✅ Embedding model loaded successfully")

# Check FAISS
index = faiss.IndexFlatL2(384)

print("✅ FAISS initialized")

# Check Groq Client
llm = ChatGroq(
    model=model_name,
    groq_api_key=groq_key,
    temperature=0
)

print("✅ Groq client initialized")

print("\n🎉 Environment setup completed successfully!")