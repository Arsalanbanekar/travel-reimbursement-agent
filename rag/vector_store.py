"""
FAISS vector store management.

Future purpose:
- Generate embeddings locally with sentence-transformers/all-MiniLM-L6-v2.
- Build, persist, and load a FAISS index from policy document chunks.
- Provide similarity search interface consumed by retriever.py.

Assignment mapping:
- Context grounding via retrieval (Section 3).
- Lightweight, local, free-tier-friendly stack (Section 2).
"""
"""
Creates and manages the FAISS vector store.

Responsibilities:
1. Load policy chunks
2. Generate embeddings
3. Build FAISS index
4. Save locally
"""

import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from rag.loader import chunk_policy

load_dotenv()

VECTOR_DB_PATH = "faiss_index"


# def get_embedding_model():
#     """Load the local embedding model."""

#     return HuggingFaceEmbeddings(
#         model_name="sentence-transformers/all-MiniLM-L6-v2"
#     )

def get_embedding_model():
    return HuggingFaceEmbeddings(
        model_name=os.getenv(
            "EMBEDDING_MODEL",
            "sentence-transformers/all-MiniLM-L6-v2"
        )
    )


def build_vector_store():
    """Create FAISS index from policy chunks."""

    documents = chunk_policy()

    embeddings = get_embedding_model()

    vector_store = FAISS.from_documents(
        documents=documents,
        embedding=embeddings,
    )

    vector_store.save_local(VECTOR_DB_PATH)

    return vector_store


def load_vector_store():
    """Load an existing FAISS index."""

    embeddings = get_embedding_model()

    return FAISS.load_local(
        VECTOR_DB_PATH,
        embeddings,
        allow_dangerous_deserialization=True,
    )


if __name__ == "__main__":

    print("Building vector store...")

    build_vector_store()

    print("✅ FAISS index created successfully!")

    print(f"Saved to: {Path(VECTOR_DB_PATH).resolve()}")