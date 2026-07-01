"""
Policy document loader and chunking.

Future purpose:
- Load data/travel_policy.md (Markdown).
- Split into semantically meaningful chunks for embedding and retrieval.
- Optionally load other rule files if referenced in policy text.

Assignment mapping:
- Context grounding input: short travel policy file (Section 4).
- Sample/mock policy documents only — no real company data (Section 2).
"""
"""
Loads and chunks the travel policy document for retrieval.

This module is responsible only for:
1. Reading the markdown policy
2. Splitting it into chunks
3. Returning LangChain Document objects

No embeddings or vector store logic belongs here.
"""

from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


POLICY_PATH = Path("data/travel_policy.md")


def load_policy():
    """Load the markdown travel policy."""

    if not POLICY_PATH.exists():
        raise FileNotFoundError(f"Policy not found: {POLICY_PATH}")

    return POLICY_PATH.read_text(encoding="utf-8")


def chunk_policy(chunk_size=500, chunk_overlap=100):
    """
    Split the policy into retrievable chunks.
    """

    policy = load_policy()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    documents = splitter.create_documents([policy])

    return documents


if __name__ == "__main__":

    docs = chunk_policy()

    print(f"Loaded {len(docs)} chunks\n")

    for i, doc in enumerate(docs[:3]):
        print("=" * 60)
        print(f"Chunk {i+1}")
        print(doc.page_content[:300])
        print()