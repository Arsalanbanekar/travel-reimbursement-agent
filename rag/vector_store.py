"""
Creates and manages the FAISS vector store over the travel policy.

Responsibilities: load policy chunks, generate embeddings, build the FAISS
index, and persist it locally.
"""

from functools import lru_cache

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from agent.config import EMBEDDING_MODEL, VECTOR_DB_PATH
from rag.loader import chunk_policy


@lru_cache(maxsize=1)
def get_embedding_model() -> HuggingFaceEmbeddings:
    """
    Load the local sentence-transformers embedding model.

    Cached because constructing it costs tens of seconds — it reads the model
    from disk and checks the Hugging Face hub. Rebuilding it per claim made
    every evaluation about 90 seconds slower than the LLM calls themselves.
    """

    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


def build_vector_store() -> FAISS:
    """Create the FAISS index from policy chunks and save it to disk."""

    vector_store = FAISS.from_documents(
        documents=chunk_policy(),
        embedding=get_embedding_model(),
    )

    vector_store.save_local(str(VECTOR_DB_PATH))

    return vector_store


@lru_cache(maxsize=1)
def load_vector_store() -> FAISS:
    """Load the persisted FAISS index, once per process."""

    return FAISS.load_local(
        str(VECTOR_DB_PATH),
        get_embedding_model(),
        # Safe here: the index is built by this repo from its own policy file.
        allow_dangerous_deserialization=True,
    )


if __name__ == "__main__":

    print("Building vector store...")

    build_vector_store()

    print(f"FAISS index created at: {VECTOR_DB_PATH}")
